from numpy import require
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torch.utils.data import TensorDataset, DataLoader
from abc import ABC
from tqdm import tqdm, trange
import numpy as np
from torch.utils.data import WeightedRandomSampler
from sklearn.decomposition import NMF
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torchbd.loss import BetaDivLoss
from scipy.stats import mannwhitneyu
import warnings

VERSION = 1.0

class dCSFA_NMF(nn.Module):
    def __init__(self,n_components,dim_in,device='auto',n_intercepts=1,n_sup_networks=1,
                optim_name='AdamW',recon_loss='IS',sup_recon_weight=1.0,sup_weight=1.0,
                useDeepEnc=True,h=256,sup_recon_type="Residual",feature_groups=None,group_weights=None,
                fixed_corr=None,momentum=0.9,sup_smoothness_weight=1):
        """
        VERSION = 1.0
        ------------------------

        Inputs
        ----------------------------
        n_components - int - The number of latent factors to learn. This is also often referred to as the number of networks
        dim_in - int - The number of features of the input. This is often the second dimension of the feature matrix. This can
                        be expressed as the number of frequencies multiplied by the number of features F
        device - {'auto','cpu','cuda'} - The device that the model will be ran on. "auto" will check for a gpu and then select it if
                                        available. Otherwise it picks cpu.
        n_intercepts - int - The number of unique intercepts to use. This allows for a different intercept for different tasks.
        n_sup_networks - int - The number of supervised networks you would like to learn. Must be less than the number of components
        optim_name - {'AdamW','Adam','SGD'} - The optimizer being used.
        recon_loss - {'IS','MSE'} - Reconstruction loss. MSE corresponds to Mean Squared Error. IS corresponds to itakura-saito.
                                    For IS loss be sure to run `pip install beta-divergence-metrics`
        sup_recon_weight - float - weight coefficient for the first component reconstruction loss.
        sup_weight - float - supervision importance weight coefficient
        useDeepEnc - bool - indicates if a neural network encoder should be used
        sup_recon_type - {"Residual","All"} - controls whether you would like to encourage the first component to reconstruct the residual of the
                                            other component reconstructions ("Residual"), or if you would like to encourage reconstruction of the
                                            whole dataset using MSE ("All")
        variational - bool - indicates if a variational autoencoder will be used.
        prior_mean - int - only used if variational. Zero is the default.
        prior_var - int - only used if variational. One is the default.
        fixed_corr - string / list - {"n/a","positive","negative"} - List where each element corresponds to the correlation constraint of a corresponding network. Also allows for string
                                            inputs of "positive" or "negative" if the number of supervised networks is 1.
        momentum - float - Momentum value if optim_name is "SGD". Default value is 0.9 per the SGD default.

        Other Member Variables
        ---------------------------
        optim_f - torch.optim.<function> - optimizer function to be used for training
        recon_loss_f - torch.optim.<function> - reconstruction loss function to be used for training
        Encoder - nn.Sequential - Single hidden layer nn.Sequential encoder that makes use of batch norm and LeakyReLU
        Encoder_A - nn.Parameter - Torch tensor for linear transformation for the linear encoder
        Encoder_b - nn.Parameter - Torch tensor for the bias in the linear encoder
        W_nmf - nn.Parameter - Input for the W_nmf decoder function that returns nn.Softplus(W_nmf)
        phi_ - nn.Parameter - Coefficient for the logistic regression classifier
        beta_ nn.Parameter - Bias vector for the logistic regression classifier
"""

        super(dCSFA_NMF,self).__init__()
        self.n_components = n_components
        self.dim_in = dim_in
        self.n_sup_networks = n_sup_networks
        self.optim_name = optim_name
        self.optim_f = self.get_optim(optim_name)
        self.recon_loss = recon_loss
        self.recon_loss_f = self.get_recon_loss(recon_loss)
        self.sup_recon_weight = sup_recon_weight
        self.sup_weight = sup_weight
        self.n_intercepts = n_intercepts
        self.useDeepEnc = useDeepEnc
        self.sup_recon_type = sup_recon_type
        self.h = h
        self.momentum = momentum
        self.sup_smoothness_weight = sup_smoothness_weight

        self.__version__ = VERSION

        if fixed_corr == None:
            self.fixed_corr = ["n/a" for sup_net in range(self.n_sup_networks)]
        elif type(fixed_corr)!=list:
            if fixed_corr.lower() == "positive":
                self.fixed_corr = ["positive"]
            elif fixed_corr.lower() == "negative":
                self.fixed_corr = ["negative"]
            elif fixed_corr.lower() == "n/a":
                self.fixed_corr = ["n/a"]
            else:
                raise ValueError("fixed corr must be a list or in {`positive`,`negative`,`n/a`}")
        else:
            assert len(fixed_corr) == len(range(self.n_sup_networks))
            self.fixed_corr = fixed_corr

        self.skl_pretrain_model = None
        self.feature_groups = feature_groups
        if feature_groups is not None and group_weights is None:
            group_weights = []
            for idx,(lb,ub) in enumerate(feature_groups):
                group_weights.append((feature_groups[-1][-1] - feature_groups[0][0])/(ub - lb))
            self.group_weights = group_weights
        else:
            self.group_weights = group_weights
            
            
        #Use deep encoder or linear
        if useDeepEnc:
            self.Encoder = nn.Sequential(nn.Linear(dim_in,self.h),
                            nn.BatchNorm1d(self.h),
                            nn.LeakyReLU(),
                            nn.Linear(self.h,n_components),
                            nn.Softplus(),
                            )
        else:
            self.Encoder_A = nn.Parameter(torch.randn(dim_in,n_components))
            self.Encoder_b = nn.Parameter(torch.randn(n_components))

        #Define nmf decoder parameter
        self.W_nmf = nn.Parameter(torch.rand(n_components,dim_in))

        #Logistic Regression Parameters
        self.phi_list = nn.ParameterList([nn.Parameter(torch.randn(1)) for sup_network in range(self.n_sup_networks)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.randn(n_intercepts,1)) for sup_network in range(self.n_sup_networks)])

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.to(self.device)
    
    @torch.no_grad()
    def get_optim(self,optim_name):
        '''
        returns a torch optimizer based on text input from the user
        '''
        if optim_name == "AdamW":
            return torch.optim.AdamW
        elif optim_name == "Adam":
            return torch.optim.Adam
        elif optim_name == "SGD":
            return torch.optim.SGD
        else:
            raise ValueError(f"{optim_name} is not supported")
    
    @torch.no_grad()
    def get_recon_loss(self,recon_loss):
        '''
        get the reconstruction loss
        '''
        if recon_loss == "MSE":
            return nn.MSELoss()
        elif recon_loss == "IS":
            return BetaDivLoss(beta=0,reduction="mean")
        else:
            raise ValueError(f"{recon_loss} is not supported")

    @staticmethod
    def inverse_softplus(x, eps=1e-5):
        '''
        Gets the inverse softplus for sklearn model pretraining
        '''
        # Calculate inverse softplus
        x_inv_softplus = np.log(np.exp(x) - (1.0 - eps))
        
        # Return inverse softplus
        return x_inv_softplus

    def get_W_nmf(self):
        W = nn.Softplus()(self.W_nmf)
        return W

    def get_phi(self,sup_net=0):
        if self.fixed_corr[sup_net] == "n/a":
            return self.phi_list[sup_net]
        elif self.fixed_corr[sup_net].lower() == "positive":
            return nn.Softplus()(self.phi_list[sup_net])
        elif self.fixed_corr[sup_net].lower() == "negative":
            return -1*nn.Softplus()(self.phi_list[sup_net])
        else:
            raise ValueError("Unsupported fixed_corr value")

    def get_weighted_recon_loss_f(self,X_pred,X_true):
        recon_loss = 0.0
        for weight,(lb,ub) in zip(self.group_weights,self.feature_groups):
            recon_loss += weight * self.recon_loss_f(X_pred[:,lb:ub],X_true[:,lb:ub])

        return recon_loss

    @torch.no_grad()
    def skl_pretrain(self,X,y,nmf_max_iter=100):
        '''
        Description
        ----------------------
        This method trains an sklearn NMF model to initialize W_nmf. First W_nmf is trained (using IS or MSE loss respectively) and we get the scores.
        W_nmf is then sorted according to the predictive ability of each of the components as gotten by the sklearn Logistic Regression. The sorted components
        are then saved.
        '''
        print("Pretraining NMF...")
        if self.recon_loss == "IS":
            skl_NMF = NMF(n_components=self.n_components,solver="mu",beta_loss="itakura-saito",init='nndsvda',max_iter=nmf_max_iter)
        else:
            skl_NMF = NMF(n_components=self.n_components,max_iter=nmf_max_iter)
        s_NMF = skl_NMF.fit_transform(X)
    
        selected_networks = []
        selected_aucs = []
        final_network_order = []
        for sup_net in range(self.n_sup_networks):

            class_auc_list = []
            pMask = y[:,sup_net].squeeze()==1
            nMask = ~pMask

            print("Identifying predictive components for network {}".format(sup_net))
            for component in tqdm(range(self.n_components)):

                s_pos = s_NMF[pMask==1,component].reshape(-1,1)
                s_neg = s_NMF[nMask==1,component].reshape(-1,1)
                U,pval = mannwhitneyu(s_pos,s_neg)
                U = U.squeeze()
                auc = U/(len(s_pos)*len(s_neg))

                class_auc_list.append(auc)

            class_auc_list = np.array(class_auc_list) 

            predictive_order = np.argsort(np.abs(class_auc_list - 0.5))[::-1]
            positive_predictive_order = np.argsort(class_auc_list)[::-1]
            negative_predictive_order = np.argsort(1-class_auc_list)[::-1]

            if len(selected_networks) > 0:
                for taken_network in selected_networks:
                    predictive_order = predictive_order[predictive_order != taken_network]
                    positive_predictive_order = positive_predictive_order[positive_predictive_order != taken_network]
                    negative_predictive_order = negative_predictive_order[negative_predictive_order != taken_network]

            if self.fixed_corr[sup_net]=="n/a":
                current_net = predictive_order[0]
                
            elif self.fixed_corr[sup_net].lower()=="positive":
                current_net = positive_predictive_order[0]

            elif self.fixed_corr[sup_net].lower()=="negative":
                current_net = negative_predictive_order[0]
            current_auc = class_auc_list[current_net.astype(int)]
            print("Selecting network: {} with auc {} for sup net {} using constraint {} correlation".format(current_net,current_auc,sup_net,self.fixed_corr[sup_net]))
            selected_networks.append(current_net)
            selected_aucs.append(selected_aucs)
            predictive_order = predictive_order[predictive_order != current_net]
            positive_predictive_order = positive_predictive_order[positive_predictive_order != current_net]
            negative_predictive_order = negative_predictive_order[negative_predictive_order != current_net]
        
        self.skl_pretrain_networks_ = selected_networks
        self.skl_pretrain_aucs_ = selected_aucs
        final_network_order = selected_networks

        for idx,_ in enumerate(predictive_order):
            final_network_order.append(predictive_order[idx])

        sorted_NMF = skl_NMF.components_[final_network_order]
        self.W_nmf.data = torch.from_numpy(self.inverse_softplus(sorted_NMF).astype(np.float32)).to(self.device)
        self.skl_pretrain_model = skl_NMF


    def encoder_pretrain(self,X,n_pre_epochs=25,batch_size=128,verbose=False,print_rate=5):
        '''
        Description
        ------------------------
        This method freezes the W_nmf parameter and trains the Encoder using only the full reconstruction loss.
        '''
        self.W_nmf.requires_grad = False
        self.pretrain_hist = []
        X = torch.Tensor(X).to(self.device)
        dset = TensorDataset(X)
        loader = DataLoader(dset,batch_size=batch_size,shuffle=True)
        pre_optimizer = self.optim_f(self.parameters(),lr=1e-3)
        epoch_iter = tqdm(range(n_pre_epochs))
        for epoch in epoch_iter:
            r_loss = 0.0
            for X_batch, in loader:
                pre_optimizer.zero_grad()
                X_recon = self.forward(X_batch,avgIntercept=True)[0]
                if self.feature_groups is not None:
                    loss_recon = self.get_weighted_recon_loss_f(X_recon,X_batch)
                else:
                    loss_recon = self.recon_loss_f(X_recon,X_batch)
                loss = loss_recon
                loss.backward()
                pre_optimizer.step()
                r_loss += loss_recon.item()
            self.pretrain_hist.append(r_loss)
            if verbose:
                epoch_iter.set_description("Pretrain Epoch: %d, Recon Loss: %0.2f"%(epoch,r_loss))
        self.W_nmf.requires_grad = True

    @torch.no_grad()
    def transform(self,X,intercept_mask=None):
        '''
        This method returns a forward pass without tracking gradients. Use this to get model reconstructions
        '''
        if intercept_mask is not None:
            assert intercept_mask.shape == (X.shape[0],self.n_intercepts)
            intercept_mask = torch.Tensor(intercept_mask).to(self.device)
        else:
            avgIntercept=True
        #Move to device
        X = torch.Tensor(X).to(self.device)
        X_recon,sup_recon_loss,y_pred_proba,s = self.forward(X,intercept_mask,avgIntercept=avgIntercept)

        return X_recon.cpu().detach().numpy(), sup_recon_loss.cpu().detach().numpy(), y_pred_proba.cpu().detach().numpy(), s.cpu().detach().numpy()

    @torch.no_grad()
    def project(self,X):
        s = self.transform(X)[3]
        return s
        
    @torch.no_grad()
    def predict(self,X,intercept_mask=None,include_scores=False):
        '''
        Returns a boolean array of predicted labels. Use include_scores=True to get
        the original scores
        '''
        if include_scores:
            y_pred,s = self.predict_proba(X,intercept_mask,include_scores)
            y_pred = y_pred >0.5
            return y_pred,s
        else:
            y_pred = self.predict_proba(X,intercept_mask,include_scores)
            return y_pred>0.5
        
    @torch.no_grad()
    def predict_proba(self,X,intercept_mask=None,include_scores=False):
        y_pred_proba,s = self.transform(X,intercept_mask)[2:]
        
        y_class = y_pred_proba
        if include_scores:
            return y_class, s
        else:
            return y_class

    def get_sup_recon(self,s):
        return s[:,self.n_sup_networks].view(-1,self.n_sup_networks) @ self.get_W_nmf()[:self.n_sup_networks,:].view(self.n_sup_networks,-1)

    def get_residual_scores(self,X,s):
        resid = (X-s[:,self.n_sup_networks:] @ self.get_W_nmf()[self.n_sup_networks:,:])
        w_sup = self.get_W_nmf()[:self.n_sup_networks,:].view(self.n_sup_networks,-1)
        s_h = resid @ w_sup.T @ torch.inverse(w_sup@w_sup.T)
        return s_h

    def residual_loss_f(self,s,s_h):
        res_loss = torch.norm(s[:,:self.n_sup_networks].view(-1,self.n_sup_networks)-s_h) / (1 - self.sup_smoothness_weight*torch.exp(-torch.norm(s_h)))
        return res_loss

    #def multiNetwork_BCE(self,y_pred,y_true,weights):
    #    mn_BCE = 0
    #    for network in range(self.n_sup_networks):
    #        mn_BCE += nn.BCELoss(weight=weights[:,network])(y_true[:,network],y_pred[:,network])
    #    print(mn_BCE)
    #    return mn_BCE

    def forward(self,X,intercept_mask=None,avgIntercept=False):
        #Encode X using the deep or linear encoder
        if self.useDeepEnc:
            s = self.Encoder(X)
        else:
            s = nn.Softplus()(X @ self.Encoder_A + self.Encoder_b)

        y_pred_list = []
        for sup_net in range(self.n_sup_networks):

            if self.n_intercepts == 1:
                y_pred_proba = nn.Sigmoid()(s[:,sup_net].view(-1,1) * self.get_phi(sup_net) + self.beta_list[sup_net]).squeeze()
            elif self.n_intercepts > 1 and not avgIntercept:
                y_pred_proba = nn.Sigmoid()(s[:,sup_net].view(-1,1) * self.get_phi(sup_net) + intercept_mask @ self.beta_list[sup_net]).squeeze()
            else:
                intercept_mask = torch.ones(X.shape[0],self.n_intercepts).to(self.device) / self.n_intercepts
                y_pred_proba = nn.Sigmoid()(s[:,sup_net].view(-1,1) * self.get_phi(sup_net) + intercept_mask @ self.beta_list[sup_net]).squeeze()

            y_pred_list.append(y_pred_proba.view(-1,1))
        
        y_pred_proba = torch.cat(y_pred_list,dim=1)
        X_recon = s @ self.get_W_nmf()

        if self.sup_recon_type == "Residual":
            s_h = self.get_residual_scores(X,s)
            sup_recon_loss = self.residual_loss_f(s,s_h)
        elif self.sup_recon_type == "All":
            X_recon = self.get_sup_recon(s)
            sup_recon_loss = self.recon_loss_f(X_recon,X)
        else:
            raise ValueError("self.sup_recon_type must be one of the following: {'Residual','All'}")
        
        return X_recon,sup_recon_loss, y_pred_proba, s

    def fit(self,X,y,y_pred_weights=None,y_sample_groups=None,intercept_mask=None,task_mask=None,n_epochs=100,n_pre_epochs=100,nmf_max_iter=100,batch_size=128,lr=1e-3, 
            pretrain=True,verbose=False,print_rate=5,X_val=None,y_val=None,y_pred_weights_val=None,task_mask_val=None,best_model_name="dCSFA-NMF-best-model.pt"):

        if intercept_mask is not None:
            assert intercept_mask.shape == (X.shape[0],self.n_intercepts)
            intercept_mask = torch.Tensor(intercept_mask).to(self.device)
        elif intercept_mask is None and self.n_intercepts==1:
            intercept_mask = torch.ones(X.shape[0],1).to(self.device)
        else:
            raise ValueError("intercept mask cannot be type None and n_intercepts greater than 1")


        #Zero out the training loss histories
        self.training_hist = []
        self.recon_hist = []
        self.recon_z1_hist = []
        self.score_reg_hist = []
        self.pred_hist = []
        if verbose: print("Pretraining....")
        if pretrain:
            self.skl_pretrain(X,y,nmf_max_iter)
            self.encoder_pretrain(X,n_pre_epochs=n_pre_epochs,verbose=verbose,
                                    print_rate=print_rate,batch_size=batch_size)
        if verbose: print("Pretraining Complete")

        #Prepare sampler information
        if y_sample_groups is None:
            y_sample_groups = y[:,0]
        
        class_sample_counts = np.array([np.sum(y_sample_groups==group) for group in np.unique(y_sample_groups)])
        weight = 1. / class_sample_counts
        samples_weights = np.array([weight[t] for t in y_sample_groups.astype(int)]).squeeze()
        samples_weights = torch.Tensor(samples_weights)
        #Send information to Tensors
        X = torch.Tensor(X).to(self.device)
        y = torch.Tensor(y).to(self.device)

        if task_mask is None:
            task_mask = torch.ones_like(y).to(self.device)
        else:
            task_mask = torch.Tensor(task_mask).to(self.device)

        if y_pred_weights is None:
            y_pred_weights = torch.ones((y[:,0].shape[0],1)).to(self.device)
        else:
            y_pred_weights = torch.Tensor(y_pred_weights).to(self.device)

        if X_val is not None and y_val is not None:
            self.best_model_name = best_model_name
            self.best_val_loss = 1e8
            self.val_loss_hist = []
            self.val_recon_loss_hist = []
            self.val_sup_recon_loss_hist = []
            self.val_pred_loss_hist = []
            X_val = torch.Tensor(X_val).to(self.device)
            y_val = torch.Tensor(y_val).to(self.device)

            if task_mask_val is None:
                task_mask_val = torch.ones_like(y_val).to(self.device)
            else:
                task_mask_val = torch.Tensor(task_mask_val).to(self.device)

            if y_pred_weights_val is None:
                y_pred_weights_val = torch.ones((y_val[:,0].shape[0],1)).to(self.device)
            else:
                y_pred_weights_val = torch.Tensor(y_pred_weights_val).to(self.device)

        dset = TensorDataset(X,y,intercept_mask,task_mask,y_pred_weights)
        sampler = WeightedRandomSampler(samples_weights,len(samples_weights))
        loader = DataLoader(dset,batch_size=batch_size,sampler=sampler)

        if self.optim_name == "SGD":
            optimizer = self.optim_f(self.parameters(),lr=lr,momentum=self.momentum)
        else:
            optimizer = self.optim_f(self.parameters(),lr=lr)

        if verbose: 
            print("Beginning Training")
            epoch_iter = tqdm(range(n_epochs))
        else:
            epoch_iter = range(n_epochs)
        for epoch in epoch_iter:
            epoch_loss = 0.0
            recon_e_loss = 0.0
            sup_recon_e_loss = 0.0
            pred_e_loss = 0.0

            for X_batch, y_batch, b_mask_batch,task_mask_batch,y_pred_weight_batch in loader:
                optimizer.zero_grad()

                X_recon, sup_recon_loss, y_pred_proba, s = self.forward(X_batch,b_mask_batch,avgIntercept=False)

                if self.feature_groups is not None:
                    recon_loss = self.get_weighted_recon_loss_f(X_recon,X_batch)
                else:
                    recon_loss = self.recon_loss_f(X_recon,X_batch)
                sup_recon_loss = self.sup_recon_weight * sup_recon_loss
                pred_loss = self.sup_weight * nn.BCELoss(weight=y_pred_weight_batch)(y_pred_proba*task_mask_batch,y_batch*task_mask_batch)
                #pred_loss = self.sup_weight * self.multiNetwork_BCE(y_pred=y_pred_proba,y_true=y_batch,weights=task_mask_batch)
                loss = recon_loss + pred_loss + sup_recon_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                recon_e_loss += recon_loss.item()
                sup_recon_e_loss += sup_recon_loss.item()
                pred_e_loss += pred_loss.item()
            
            self.training_hist.append(epoch_loss)

            #Get epoch training data performance
            with torch.no_grad():
                X_recon_train,sup_recon_loss_train,y_pred_proba_train,_ = self.forward(X,intercept_mask)
                training_mse_loss = nn.MSELoss()(X_recon_train,X).item()
                training_sample_auc_list = []
                for sup_net in range(self.n_sup_networks):
                    training_sample_auc_list.append(roc_auc_score(y.cpu().detach().numpy()[:,sup_net],y_pred_proba_train.cpu().detach().numpy()[:,sup_net]))

                self.recon_hist.append(training_mse_loss)
                self.pred_hist.append(training_sample_auc_list)

            #Evaluate Validation Performance
            if X_val is not None and y_val is not None:
                with torch.no_grad():
                    X_recon_val,sup_recon_loss_val,y_pred_proba_val,_ = self.forward(X_val,avgIntercept=True)
                    if self.feature_groups is not None:
                        val_recon_loss = self.get_weighted_recon_loss_f(X_recon_val,X_val)
                    else:
                        val_recon_loss = self.recon_loss_f(X_recon_val,X_val)
                    val_sup_recon_loss = self.sup_recon_weight * sup_recon_loss_val
                    val_pred_loss = self.sup_weight * nn.BCELoss(y_pred_weights_val)(y_pred_proba_val*task_mask_val,y_val*task_mask_val)
                    #val_pred_loss = self.sup_weight * self.multiNetwork_BCE(y_pred=y_pred_proba_val,y_true=y_val,weights=task_mask_val)
                    val_loss = val_recon_loss+val_sup_recon_loss+val_pred_loss

                    #val AUC
                    val_mse = nn.MSELoss()(X_val,X_recon_val).item()
                    val_sample_auc_list = []
                    for sup_net in range(self.n_sup_networks):
                        val_sample_auc_list.append(roc_auc_score(y_val.cpu().detach().numpy()[:,sup_net],y_pred_proba_val.cpu().detach().numpy()[:,sup_net]))

                    self.val_loss_hist.append(val_loss.item())
                    self.val_recon_loss_hist.append(val_recon_loss.item())
                    self.val_sup_recon_loss_hist.append(val_sup_recon_loss.item())
                    self.val_pred_loss_hist.append(val_sample_auc_list)
                
                if val_loss.item() < self.best_val_loss:
                    self.best_epoch = epoch
                    self.best_val_recon = val_recon_loss
                    self.best_val_auc = val_sample_auc_list
                    self.best_val_loss = val_loss.item()
                    torch.save(self.state_dict(),self.best_model_name)

            if verbose and (X_val is not None and y_val is not None):
                epoch_iter.set_description("Epoch: {}, Best Epoch: {} Best Val Recon: {}, Best Val by Window ROC-AUC: {} loss: {}, recon: {}, pred by Window roc-auc: {}".format(epoch,self.best_epoch,self.best_val_recon,self.best_val_auc,
                                                                                                                                                                    epoch_loss, training_mse_loss,training_sample_auc_list))
            elif verbose:
                epoch_iter.set_description("Epoch: {}, loss: {:.2}, recon: {:.2}, sample pred by Window roc-aucs: {}".format(epoch,epoch_loss,training_mse_loss,training_sample_auc_list))
        
        if X_val is not None and y_val is not None:
            print("Loading best model...")
            self.load_state_dict(torch.load(self.best_model_name))
            print("Done!")

    def _component_recon(self,h,component):
        W = self.get_W_nmf()
        X_recon = h[:,component].view(-1,1) @ W[component,:].view(1,-1)
        return X_recon
    
    @torch.no_grad()
    def get_comp_recon(self,h,component):
        h = torch.Tensor(h).to(self.device)
        X_recon = self._component_recon(h,component)
        return X_recon.cpu().detach().numpy()

    @torch.no_grad()
    def get_skl_mse_score(self,X,nmf_max_iter=500):
        if self.skl_pretrain_model is not None:
            s_skl = self.skl_pretrain_model.transform(X)
        else:
            warnings.warn("No Pretraining NMF model present - Training new one from scratch")
            if self.recon_loss == "IS":
                skl_NMF = NMF(n_components=self.n_components,solver="mu",beta_loss="itakura-saito",init='nndsvda',nmf_max_iter=100)
            else:
                skl_NMF = NMF(n_components=self.n_components,nmf_max_iter=100)
            s_skl = skl_NMF.fit_transform(X)
            self.skl_pretrain_model = skl_NMF
        X_recon_skl = s_skl @ self.skl_pretrain_model.components_
        skl_mse = np.mean((X_recon_skl-X)**2)
        return skl_mse

    @torch.no_grad()
    def get_mse_score(self,X):
        X_recon_nn = self.transform(X)[0]
        nn_mse = np.mean((X_recon_nn-X)**2)

        return nn_mse

    @torch.no_grad()
    def get_skl_nn_mse(self,X):
        "Returns the sklearn mse and dCSFA mse"
        skl_mse = self.get_skl_mse_score(X)
        nn_mse = self.get_mse_score(X)
        return skl_mse, nn_mse