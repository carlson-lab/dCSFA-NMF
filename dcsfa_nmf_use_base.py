from numpy import require
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
import numpy as np
from nmf_base_class import NMF_Base
from torch.utils.data import WeightedRandomSampler
from sklearn.decomposition import NMF
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torchbd.loss import BetaDivLoss
from scipy.stats import mannwhitneyu
import warnings

class dCSFA_NMF(NMF_Base):
    def __init__(self,n_components,device='auto',n_intercepts=1,n_sup_networks=1,
                optim_name='AdamW',recon_loss='MSE',recon_weight=1.0,sup_weight=1.0,sup_recon_weight=1.0,
                useDeepEnc=True,h=256,sup_recon_type="Residual",feature_groups=None,group_weights=None,
                fixed_corr=None,momentum=0.9,lr=1e-3,sup_smoothness_weight=1.0,saveFolder="~",verbose=False):

        super(dCSFA_NMF,self).__init__(n_components,device,n_sup_networks,fixed_corr,recon_loss,recon_weight,
                sup_recon_type,sup_recon_weight,sup_smoothness_weight,feature_groups,group_weights,verbose)

        self.n_intercepts = n_intercepts
        self.optim_name=optim_name
        self.optim_alg = self.get_optim(optim_name) #definition in nmf_base_class
        self.pred_loss_f = nn.BCELoss
        self.recon_weight = recon_weight
        self.sup_weight = sup_weight
        self.useDeepEnc = useDeepEnc
        self.h = h
        self.momentum = momentum
        self.lr = lr
        self.sup_smoothness_weight = sup_smoothness_weight
        self.saveFolder = saveFolder
        print("Model parameters will be saved to directory: {}".format(saveFolder))

    def _initialize(self,dim_in):
        self.dim_in = dim_in

        self._initialize_NMF(dim_in) #definition in nmf_base_class

        if self.useDeepEnc:
            self.Encoder = nn.Sequential(nn.Linear(dim_in,self.h),
                            nn.BatchNorm1d(self.h),
                            nn.LeakyReLU(),
                            nn.Linear(self.h,self.n_components),
                            nn.Softplus(),
                            )
        
        else:
            self.Encoder_A = nn.Parameter(torch.randn(dim_in,self.n_components))
            self.Encoder_b = nn.Parameter(torch.randn(self.n_components))

        #Logistic Regression Parameters
        self.phi_list = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(self.n_sup_networks)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.randn(self.n_intercepts,1)) for _ in range(self.n_sup_networks)])

        self.to(self.device) #device set in nmf_base_class

    def instantiate_optimizer(self):
        if self.optim_name == "SGD":
            optimizer = self.optim_alg(self.parameters(),lr=self.lr,momentum=self.momentum)
        else:
            optimizer = self.optim_alg(self.parameters(),lr=self.lr)
        return optimizer

    def get_all_class_predictions(self,X,s,intercept_mask,avgIntercept):

        if intercept_mask is None and avgIntercept is False:
            warnings.warn("Intercept mask cannot be none and avgIntercept False... Averaging Intercepts")
            avgIntercept=True

        #Get predictions for each class
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

        #Concatenate predictions into a single matrix [n_samples,n_tasks]
        y_pred_proba = torch.cat(y_pred_list,dim=1)

        return y_pred_proba

    def get_embedding(self,X):
        #Encode X using the deep or linear encoder
        if self.useDeepEnc:
            s = self.Encoder(X)
        else:
            s = nn.Softplus()(X @ self.Encoder_A + self.Encoder_b)
        
        return s

    def get_phi(self,sup_net=0):
        if self.fixed_corr[sup_net] == "n/a":
            return self.phi_list[sup_net]
        elif self.fixed_corr[sup_net].lower() == "positive":
            return nn.Softplus()(self.phi_list[sup_net])
        elif self.fixed_corr[sup_net].lower() == "negative":
            return -1*nn.Softplus()(self.phi_list[sup_net])
        else:
            raise ValueError("Unsupported fixed_corr value")


    def forward(self,X,y,task_mask,pred_weight,intercept_mask=None,avgIntercept=False):

        #Get the scores from the encoder
        s = self.get_embedding(X)

        #Get the reconstruction losses
        recon_loss = self.NMF_decoder_forward(X,s)

        #Get predictions
        y_pred = self.get_all_class_predictions(X,s,intercept_mask,avgIntercept)
        pred_loss = self.sup_weight*self.pred_loss_f(weight=pred_weight)(y_pred*task_mask,y*task_mask)

        #recon loss and pred loss are left seperate so only recon_loss can be applied for
        #encoder pretraining
        return recon_loss, pred_loss

    @torch.no_grad()
    def transform(self,X,intercept_mask=None,avgIntercept=True,return_npy=True):

        if not torch.is_tensor(X):
            X = torch.Tensor(X).float().to(self.device)

        s = self.get_embedding(X)
        X_recon = self.get_all_comp_recon(s)
        y_pred = self.get_all_class_predictions(X,s,intercept_mask,avgIntercept)

        if return_npy:
            s = s.detach().cpu().numpy()
            X_recon = X_recon.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
        
        return X_recon, y_pred, s


    def pretrain_encoder(self,X,y,y_pred_weights,task_mask,intercept_mask,sample_weights,n_pre_epochs=100,batch_size=128):

        #Freeze the decoder
        self.W_nmf.requires_grad = False

        #Load arguments onto device
        X = torch.Tensor(X).float().to(self.device)
        y = torch.Tensor(y).float().to(self.device)
        y_pred_weights = torch.Tensor(y_pred_weights).float().to(self.device)
        task_mask = torch.Tensor(task_mask).long().to(self.device)
        intercept_mask = torch.Tensor(intercept_mask).to(self.device)
        sample_weights = torch.Tensor(sample_weights).to(self.device)

        #create a dset
        dset = TensorDataset(X,y,y_pred_weights,task_mask,intercept_mask)
        sampler = WeightedRandomSampler(sample_weights,len(sample_weights))
        loader = DataLoader(dset,batch_size=batch_size,sampler=sampler)

        #Instantiate Optimizer
        optimizer = self.instantiate_optimizer()

        #Define iterator
        if self.verbose:
            epoch_iter = tqdm(range(n_pre_epochs))
        else:
            epoch_iter = range(n_pre_epochs)

        for epoch in epoch_iter:
            r_loss = 0.0
            for X_batch,y_batch,y_pred_weights_batch,task_mask_batch,intercept_mask_batch in loader:
                optimizer.zero_grad()
                recon_loss,_ = self.forward(X_batch,y_batch,task_mask_batch,y_pred_weights_batch,intercept_mask_batch)
                recon_loss.backward()
                optimizer.step()
                r_loss += recon_loss.item()
            
            if self.verbose:
                epoch_iter.set_description("Encoder Pretrain Epoch: {}, Recon Loss: {:.6}".format(epoch,r_loss/len(loader)))

        #Unfreeze the NMF decoder
        self.W_nmf.requires_grad = True

    def fit(self,X,y,y_pred_weights=None,task_mask=None,intercept_mask=None,y_sample_groups=None,n_epochs=100,n_pre_epochs=100,nmf_max_iter=100,
            batch_size=128,lr=1e-3,pretrain=True,verbose=False,X_val=None,y_val=None,y_pred_weights_val=None,task_mask_val=None,
            best_model_name="dCSFA-NMF-best-model.pt"):

        #Initialize model parameters
        self._initialize(X.shape[1])

        #establish loss histories
        self.training_hist = [] #tracks average overall loss value during training
        self.recon_hist = [] #tracks training data mse
        self.pred_hist = [] #tracks training data aucs

        #Globaly activate/deactivate print statements
        self.verbose=verbose
        self.lr = lr

        #Fill default values
        if intercept_mask is None:
            intercept_mask = np.ones((X.shape[0],self.n_intercepts))

        if task_mask is None:
            task_mask = np.ones(y.shape)

        if y_pred_weights is None:
            y_pred_weights = np.ones((y.shape[0],1))

        #Fill sampler parameters
        if y_sample_groups is None:
            y_sample_groups = np.ones((y.shape[0]))
            samples_weights = y_sample_groups
        else:
            class_sample_counts = np.array([np.sum(y_sample_groups==group) for group in np.unique(y_sample_groups)])
            weight = 1. / class_sample_counts
            samples_weights = np.array([weight[t] for t in y_sample_groups.astype(int)]).squeeze()
            samples_weights = torch.Tensor(samples_weights)

        #Pretrain the model
        if pretrain:
            self.pretrain_NMF(X,y,nmf_max_iter)
            self.pretrain_encoder(X,y,y_pred_weights,task_mask,intercept_mask,samples_weights,n_pre_epochs,batch_size)
            
        #Send Training Arguments to Tensors
        X = torch.Tensor(X).float().to(self.device)
        y = torch.Tensor(y).float().to(self.device)
        y_pred_weights = torch.Tensor(y_pred_weights).float().to(self.device)
        task_mask = torch.Tensor(task_mask).long().to(self.device)
        intercept_mask = torch.Tensor(intercept_mask).long().to(self.device)
        samples_weights = torch.Tensor(samples_weights).to(self.device)

        #If validation data is provided, set up the tensors
        if X_val is not None and y_val is not None:
            assert best_model_name.split('.')[-1] == "pt", f"Save file `{self.saveFolder + best_model_name}` must be of type .pt"
            self.best_model_name = best_model_name
            self.best_val_recon = 1e8
            self.best_val_avg_auc = 0.0
            self.val_recon_hist = []
            self.val_pred_hist = []

            if task_mask_val is None:
                task_mask_val = np.ones(y_val.shape)

            if y_pred_weights_val is None:
                y_pred_weights_val = np.ones((y_val[:,0].shape[0],1))

            X_val = torch.Tensor(X_val).float().to(self.device)
            y_val = torch.Tensor(y_val).float().to(self.device)
            task_mask_val = torch.Tensor(task_mask_val).long().to(self.device)
            y_pred_weights_val = torch.Tensor(y_pred_weights_val).float().to(self.device)


        #Instantiate the dataloader and optimizer
        dset = TensorDataset(X,y,y_pred_weights,task_mask,intercept_mask)
        sampler = WeightedRandomSampler(samples_weights,len(samples_weights))
        loader = DataLoader(dset,batch_size=batch_size,sampler=sampler)
        optimizer = self.instantiate_optimizer()

        #Define Training Iterator
        if self.verbose: 
            print("Beginning Training")
            epoch_iter = tqdm(range(n_epochs))
        else:
            epoch_iter = range(n_epochs)

        #Training Loop
        for epoch in epoch_iter:
            epoch_loss = 0.0
            recon_e_loss = 0.0
            pred_e_loss = 0.0

            for X_batch,y_batch,y_pred_weights_batch,task_mask_batch,intercept_mask_batch in loader:

                self.train()
                optimizer.zero_grad()
                recon_loss, pred_loss = self.forward(X_batch,y_batch,task_mask_batch,y_pred_weights_batch,intercept_mask_batch)
                loss = recon_loss + pred_loss #Weighting happens inside of the forward call
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                recon_e_loss += recon_loss.item()
                pred_e_loss += pred_loss.item()

            self.training_hist.append(epoch_loss / len(loader))
            with torch.no_grad():
                self.eval()
                X_recon, y_pred, s = self.transform(X,y,intercept_mask,return_npy=False)
                training_mse_loss = nn.MSELoss()(X_recon,X).item()
                training_auc_list = []
                for sup_net in range(self.n_sup_networks):
                    auc = roc_auc_score(y.detach().cpu().numpy()[task_mask[:,sup_net].detach().cpu().numpy()==1,sup_net],
                                        y_pred.detach().cpu().numpy()[task_mask[:,sup_net].detach().cpu().numpy()==1,sup_net])
                    training_auc_list.append(auc)
                
                self.recon_hist.append(training_mse_loss)
                self.pred_hist.append(training_auc_list)

                #If Validation data is present, collect performance metrics
                if X_val is not None and y_val is not None:
                    X_recon_val,y_pred_val,s_val = self.transform(X_val,y_val,return_npy=False)
                    validation_mse_loss = nn.MSELoss()(X_recon_val,X_val).item()
                    validation_auc_list = []
                    for sup_net in range(self.n_sup_networks):
                        auc = roc_auc_score(y_val.detach().cpu().numpy()[task_mask_val[:,sup_net].detach().cpu().numpy()==1,sup_net],
                                            y_pred_val.detach().cpu().numpy()[task_mask_val[:,sup_net].detach().cpu().numpy()==1,sup_net])
                        validation_auc_list.append(auc)
                    
                    self.val_recon_hist.append(validation_mse_loss)
                    self.val_pred_hist.append(validation_auc_list)

                    if validation_mse_loss < self.best_val_recon and np.mean(validation_auc_list) > self.best_val_avg_auc:
                        self.best_epoch = epoch
                        self.best_val_recon = validation_mse_loss
                        self.best_val_aucs = validation_auc_list
                        torch.save(self.state_dict(),self.saveFolder + self.best_model_name)

                    if self.verbose:
                        epoch_iter.set_description("Epoch: {}, Best Epoch: {}, Best Val MSE: {:.6}, Best Val by Window ROC-AUC {}, current MSE: {:.6}, current AUC: {}".format(epoch,
                                                                                                                                                                        self.best_epoch,
                                                                                                                                                                        self.best_val_recon,
                                                                                                                                                                        self.best_val_aucs,
                                                                                                                                                                        validation_mse_loss,
                                                                                                                                                                        validation_auc_list))
                else:
                    epoch_iter.set_description("Epoch: {}, Current Training MSE: {:.6}, Current Training by Window ROC-AUC: {}".format(epoch,training_mse_loss,training_auc_list))
        
        if X_val is not None and y_val is not None:
            if self.verbose:
                print("Loaded the best model from Epoch: {} with MSE: {:.6} and AUCs: {}".format(self.best_epoch,self.best_val_recon,self.best_val_aucs))
            self.load_state_dict(torch.load(self.saveFolder+self.best_model_name))

    def reconstruct(self,X,component=None):
        X_recon,_,s = self.transform(X)

        if component is not None:
            X_recon = self.get_comp_recon(s,component)

        return X_recon

    def predict_proba(self,X,return_scores=False):
        _,y_pred,s = self.transform(X)

        if return_scores:
            return y_pred, s
        else:
            return y_pred

    def predict(self,X,return_scores=False):

        _,y_pred,s = self.transform(X)

        if return_scores:
            return y_pred > 0.5, s
        else:
            return y_pred > 0.5

    def project(self,X):
        _,_,s = self.transform(X)
        return s

    def score(self,X,y,groups=None,return_dict=False):
        _,y_pred,_ = self.transform(X)

        if groups is not None:
            auc_dict = {}
            for group in np.unique(groups):
                auc_list = []
                for sup_net in range(self.n_sup_networks):
                    auc = roc_auc_score(y[:,sup_net],y_pred[:,sup_net])
                    auc_list.append(auc)
                
                auc_dict[group] = auc_list

            if return_dict:
                return auc_dict

            else:
                auc_array = np.vstack([auc_dict[key] for key in np.unique(groups)])
                return np.mean(auc_array,axis=0)
        
        else:
            auc_list = []
            for sup_net in range(self.n_sup_networks):
                auc = roc_auc_score(y[:,sup_net],y_pred[:,sup_net])
                auc_list.append(auc)

            return np.array(auc_list)





            







        