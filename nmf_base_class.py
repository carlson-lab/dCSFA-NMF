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


class NMF_Base(nn.Module):

    def __init__(self,n_components,device='auto',n_sup_networks=1,fixed_corr=None,recon_loss="MSE",recon_weight=1.0,sup_recon_type="Residual",
                sup_recon_weight=1.0,sup_smoothness_weight=1,feature_groups=None,group_weights=None,verbose=False):
        super(NMF_Base,self).__init__()

        self.n_components = n_components
        self.n_sup_networks = n_sup_networks
        self.recon_loss = recon_loss
        self.recon_weight = recon_weight
        self.sup_recon_type = sup_recon_type
        self.sup_smoothness_weight = sup_smoothness_weight
        self.sup_recon_weight = sup_recon_weight
        self.verbose = verbose

        self.recon_loss_f = self.get_recon_loss(recon_loss)

        #Set correlation constraints
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

        self.feature_groups = feature_groups
        if feature_groups is not None and group_weights is None:
            group_weights = []
            for idx,(lb,ub) in enumerate(feature_groups):
                group_weights.append((feature_groups[-1][-1] - feature_groups[0][0])/(ub - lb))
            self.group_weights = group_weights
        else:
            self.group_weights = group_weights

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def _initialize_NMF(self,dim_in):
        """
        Instantiates the NMF decoder and moves the NMF_Base instance to self.device

        Parameters
        ----------
        dim_in : int
            Number of total features
        """
        self.W_nmf = nn.Parameter(torch.rand(self.n_components,dim_in))
        self.to(self.device)

    @staticmethod
    def inverse_softplus(x, eps=1e-5):
        '''
        Gets the inverse softplus for sklearn model pretraining
        '''
        # Calculate inverse softplus
        x_inv_softplus = np.log(np.exp(x+eps) - (1.0 - eps))
        
        # Return inverse softplus
        return x_inv_softplus

    def get_W_nmf(self):
        """
        Passes the W_nmf parameter through a softplus function to make it non-negative
        """
        return nn.Softplus()(self.W_nmf)

    @torch.no_grad()
    def get_recon_loss(self,recon_loss):
        """
        Returns the reconstruction loss function

        Parameters
        ----------
        recon_loss : str in {"MSE","IS"}
            Identifies which loss function to use
        """
        if recon_loss == "MSE":
            return nn.MSELoss()
        elif recon_loss == "IS":
            return BetaDivLoss(beta=0,reduction="mean")
        else:
            raise ValueError(f"{recon_loss} is not supported")

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
    def pretrain_NMF(self,X,y,nmf_max_iter=100):
        """
        Trains an unsupervised NMF model and sorts components by predictiveness for corresponding tasks.
        Saved NMF components are stored as a member variable self.W_nmf. Sklearn NMF model is saved as
        NMF_init.

        Parameters
        ----------
        X : numpy.ndarray
            Input features
            Shape: ``[n_samps,n_features]``
        
        y : numpy.ndarray
            Input labels
            Shape: ``[n_samps,n_sup_networks]``

        nmf_max_iter : int
            Maximum iterations for convergence using sklearn.decomposition.NMF
        
        Returns
        ---------

        None
        """

        if self.verbose: print("Pretraining NMF...")

        # Initialize the model - solver corresponds to defined recon loss
        if self.recon_loss == "IS":
            self.NMF_init = NMF(n_components=self.n_components,solver="mu",beta_loss="itakura-saito",init='nndsvda',max_iter=nmf_max_iter)
        else:
            self.NMF_init = NMF(n_components=self.n_components,max_iter=nmf_max_iter,init="nndsvd")

        #Fit the model
        s_NMF = self.NMF_init.fit_transform(X)

        #define arrays for storing model predictive information
        selected_networks = []
        selected_aucs = []
        final_network_order = []

        #Find the most predictive component for each task - the first task gets priority
        for sup_net in range(self.n_sup_networks):
            
            #Prep labels and set auc storage
            class_auc_list = []

            #Create progress bar if verbose
            if self.verbose:
                print("Identifying predictive components for supervised network {}".format(sup_net))
                component_iter = tqdm(range(self.n_components))
            else:
                component_iter = range(self.n_components)

            #Find each components AUC for the current task
            for component in component_iter:
                s_pos = s_NMF[y[:,sup_net]==1,component].reshape(-1,1)
                s_neg = s_NMF[y[:,sup_net]==0,component].reshape(-1,1)
                U,_ = mannwhitneyu(s_pos,s_neg)
                U = U.squeeze()
                auc = U/(len(s_pos)*len(s_neg))

                class_auc_list.append(auc)

            class_auc_list = np.array(class_auc_list) 

            #Sort AUC predictions based on correlations
            predictive_order = np.argsort(np.abs(class_auc_list - 0.5))[::-1]
            positive_predictive_order = np.argsort(class_auc_list)[::-1]
            negative_predictive_order = np.argsort(1-class_auc_list)[::-1]

            #Ignore components that have been used for previous supervised tasks
            if len(selected_networks) > 0:
                for taken_network in selected_networks:
                    predictive_order = predictive_order[predictive_order != taken_network]
                    positive_predictive_order = positive_predictive_order[positive_predictive_order != taken_network]
                    negative_predictive_order = negative_predictive_order[negative_predictive_order != taken_network]

            #Select the component based on predictive correlation
            if self.fixed_corr[sup_net]=="n/a":
                current_net = predictive_order[0]
                
            elif self.fixed_corr[sup_net].lower()=="positive":
                current_net = positive_predictive_order[0]

            elif self.fixed_corr[sup_net].lower()=="negative":
                current_net = negative_predictive_order[0]

            current_auc = class_auc_list[current_net.astype(int)]

            #Declare the selected network and save the network and chosen AUCs
            if self.verbose:
                print("Selecting network: {} with auc {} for sup net {} using constraint {} correlation".format(current_net,current_auc,sup_net,self.fixed_corr[sup_net]))
            selected_networks.append(current_net)
            selected_aucs.append(selected_aucs)
            predictive_order = predictive_order[predictive_order != current_net]
            positive_predictive_order = positive_predictive_order[positive_predictive_order != current_net]
            negative_predictive_order = negative_predictive_order[negative_predictive_order != current_net]
        
        #Save the selected networks and corresponding AUCs
        self.skl_pretrain_networks_ = selected_networks
        self.skl_pretrain_aucs_ = selected_aucs
        final_network_order = selected_networks

        #Get the final sorting of the networks for predictiveness
        for idx,_ in enumerate(predictive_order):
            final_network_order.append(predictive_order[idx])

        #Save the final sorted components to the W_nmf parameter
        sorted_NMF = self.NMF_init.components_[final_network_order]
        self.W_nmf.data = torch.from_numpy(self.inverse_softplus(sorted_NMF.astype(np.float32))).to(self.device)

    def get_sup_recon(self,s):
        """
        Returns the reconstruction of all of the supervised networks

        Parameters
        ----------
        s : torch.Tensor.float()
            Factor activation scores
            Shape: ``[n_samps,n_components]``

        Returns
        -------
        X_sup_recon : torch.Tensor.float()
            Reconstruction using only supervised components
            Shape: ``[n_samps,n_features]``
        """
        X_sup_recon = s[:,:self.n_sup_networks].view(-1,self.n_sup_networks) @ self.get_W_nmf()[:self.n_sup_networks,:].view(self.n_sup_networks,-1)
        return X_sup_recon

    def get_residual_scores(self,X,s):
        """
        Returns the supervised score values that would maximize reconstruction performance based
        on the residual reconstruction.

        s_h = (X - s_unsup @ W_unsup) @ w_sup.T @ (w_sup @ w_sup.T)^(-1)

        Parameters
        ----------
        X : torch.Tensor
            Ground truth features
            Shape: ``[n_samples,n_features]``
        
        s : torch.Tensor
            Factor activation scores
            Shape: ``[n_samples,n_components]``
        """
        resid = (X-s[:,self.n_sup_networks:] @ self.get_W_nmf()[self.n_sup_networks:,:])
        w_sup = self.get_W_nmf()[:self.n_sup_networks,:].view(self.n_sup_networks,-1)
        s_h = resid @ w_sup.T @ torch.inverse(w_sup@w_sup.T)
        return s_h

    def residual_loss_f(self,s,s_h):
        """
        Loss function between supervised factor scores and the maximal values for
        reconstruction. Factors scores are encouraged to be non-zero by the smoothness weight.

        f(s,s_h) = ||s_sup - s_h||^2 / (1 - smoothness_weight * exp(-||s_h||^2))

        Parameters
        ----------
        s : torch.Tensor
            Factor activation scores
            Shape: ``[n_samples,n_components]``

        s_h : torch.Tensor
            Factor scores that would minimize the reconstruction loss
            Shape: ``[n_samples,n_components]``
        
        Returns
        ---------
        res_loss : torch.Tensor
            Residual scores loss
        """
        res_loss = torch.norm(s[:,:self.n_sup_networks].view(-1,self.n_sup_networks)-s_h) / (1 - self.sup_smoothness_weight*torch.exp(-torch.norm(s_h)))
        return res_loss

    def get_weighted_recon_loss_f(self,X_pred,X_true):
        """
        Model training often involves multiple feature types such as Power and Directed Spectrum that
        have vastly different feature counts (power: n_roi*n_freq, ds: n_roi*(n_roi-1)*n_freq).

        This loss reweights the reconstruction of each feature group proportionally to the number of features
        such that each feature type has roughly equal importance to the reconstruction.

        Parameters
        ----------
        X_pred : torch.Tensor
            Reconstructed features
            Shape: ``[n_samps,n_features]``

        X_true : torch.Tensor
            Ground Truth Features
            Shape: ``[n_samps,n_features]``

        Returns
        ---------
        recon_loss : torch.Tensor
            Weighted reconstruction loss for each feature
        """
        recon_loss = 0.0
        for weight,(lb,ub) in zip(self.group_weights,self.feature_groups):
            recon_loss += weight * self.recon_loss_f(X_pred[:,lb:ub],X_true[:,lb:ub])

        return recon_loss

    def eval_recon_loss(self,X_pred,X_true):
        """
        If using feature groups, returns weighted recon loss
        Else, returns unweighted recon loss

        Parameters
        ----------
        X_pred : torch.Tensor
            Reconstructed features
            Shape: ``[n_samps,n_features]``

        X_true : torch.Tensor
            Ground Truth Features
            Shape: ``[n_samps,n_features]``

        Returns
        ---------
        recon_loss : torch.Tensor
            Weighted or Unweighted recon loss
        """
        if self.feature_groups is None:
            recon_loss = self.recon_loss_f(X_pred,X_true)
        else:
            recon_loss = self.get_weighted_recon_loss_f(X_pred,X_true)

        return recon_loss

    def NMF_decoder_forward(self,X,s):
        """
        NMF Decoder forward pass

        Parameters
        ----------
        X : torch.Tensor
            Input Features
            Shape: ``[n_samps,n_features]``

        s : torch.Tensor
            Encoder embeddings / factor score activations
            Shape: ``[n_samps,n_components]``

        Returns
        -------
        recon_loss : torch.Tensor
            Whole data recon loss + supervised recon loss
        """
        recon_loss = 0.0

        X_recon = s @ self.get_W_nmf()

        recon_loss += self.recon_weight*self.eval_recon_loss(X_recon,X)

        if self.sup_recon_type == "Residual":
            s_h = self.get_residual_scores(X,s)
            sup_recon_loss = self.residual_loss_f(s,s_h)
        elif self.sup_recon_type == "All":
            X_recon = self.get_sup_recon(s)
            sup_recon_loss = self.recon_loss_f(X_recon,X)

        recon_loss += self.sup_recon_weight*sup_recon_loss

        return recon_loss

    @torch.no_grad()
    def get_comp_recon(self,s,component):
        """
        Gets the reconstruction for a specific component

        Parameters
        ----------
        s : torch.Tensor
            Encoder embeddings / factor score activations
            Shape: ``[n_samps,n_components]``

        component : int, 0 <= component < n_components
            Component to use for the reconstruction

        Returns
        ---------
        X_recon : numpy.ndarray
            Reconstruction using a specific component
        """
        X_recon = s[:,component].view(-1,1) @ self.get_W_nmf()[component,:].view(1,-1)
        return X_recon.detach().cpu().numpy()

    @torch.no_grad()
    def get_all_comp_recon(self,s):
        """
        Gets the reconstruction for all components

        Parameters
        ----------
        s : torch.Tensor
            Encoder embeddings / factor score activations
            Shape: ``[n_samps,n_components]``

        Returns
        ---------
        X_recon : numpy.ndarray
            Reconstruction using a specific component
        """
        X_recon = s @ self.get_W_nmf()
        return X_recon
    @torch.no_grad()
    def get_factor(self,component):
        """
        Returns the numpy array for the corresponding factor

        Parameters
        ----------
        component : int, 0 <= component < n_components
            Component to use for the reconstruction

        Returns
        ---------
        factor : np.ndarray
            Factor from W_nmf
        """
        return self.get_W_nmf()[component,:].detach().cpu().numpy()
