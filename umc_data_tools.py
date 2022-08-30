import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc,roc_auc_score
from scipy.stats import mannwhitneyu
from sklearn.model_selection import KFold
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#This Name Corrector Dictionary May need to be stored as a .json file if it gets too long
Name_Corrector_Dict = {
    'Amy':'Amy',
    'BLA':'Amy',
    'Cg_Cx':'Cg_Cx',
    'Hipp':'Hipp',
    'lDHip':'Hipp',
    'mDHip':'Hipp',
    'lSNC':'lSNC',
    'mSNC':'mSNC',
    'IL_Cx':'IL_Cx',
    'MD_Thal':'Thal',
    'Md_Thal':'Thal',
    'Thal':'Thal',
    'NAc':'Nac',
    'Nac':'Nac',
    'Acb_Core':'Nac',
    'Acb_Sh':'Nac',
    'PrL_Cx':'PrL_Cx',
    'VTA':'VTA'
}
def correct_powerFeatures(powerFeatures):
    corrected_pF = []
    for feature in powerFeatures:
        split_feature = feature.split(' ')
        split_feature[0] = Name_Corrector_Dict[split_feature[0]]
        correct_feature = " ".join(split_feature)
        corrected_pF.append(correct_feature)
    return corrected_pF

#I think this would be more flexible as a class instead of a function, but I haven't gotten to it yet
#It also doesn't handle averaging areas, For regions that would be averaged together, only one of them
#will be used in the current setup
#I think this would be more flexible as a class instead of a function, but I haven't gotten to it yet
#It also doesn't handle averaging areas, For regions that would be averaged together, only one of them
#will be used in the current setup
def augment_powerFeatures(list_of_data: list, list_of_powerFeatures: list,num_freqs=56) -> list:
    corrected_list_of_powerFeatures = [correct_powerFeatures(powerFeatures)
                                        for powerFeatures in list_of_powerFeatures]
    all_features = np.hstack(corrected_list_of_powerFeatures)
    #Get all of the unique features after naming corrections
    unique_features = np.unique(all_features)

    print(unique_features[0].shape)
    #Correct the ordering of frequencies
    unique_power_features = [" ".join([feature_name.split(" ")[0],str(freq)])
                                for feature_name in unique_features[::num_freqs]
                                for freq in range(1,num_freqs+1)]

    #Store the list of unique datasets
    augmented_list_of_data = []
    augmented_list_of_masks = []

    for data,features in zip(list_of_data,corrected_list_of_powerFeatures):
        augmented_data = np.zeros((data.shape[0],len(unique_power_features)))
        mask = np.zeros_like(augmented_data)

        for idx, feature in enumerate(features):
            f_idx = unique_power_features.index(feature)
            augmented_data[:,f_idx] = data[:,idx]
            mask[:,f_idx] = 1

        augmented_list_of_data.append(augmented_data)
        augmented_list_of_masks.append(mask)

    return augmented_list_of_data,augmented_list_of_masks,unique_power_features

def correct_cohFeatures(cohFeatures):
    corrected_cohF = []
    for feature in cohFeatures:
        split_feature = feature.split(' ')
        split_brain_regions = split_feature[0].split('-')

        split_brain_regions[0] = Name_Corrector_Dict[split_brain_regions[0]]
        split_brain_regions[1] = Name_Corrector_Dict[split_brain_regions[1]]

        corrected_brain_regions = "-".join(sorted(split_brain_regions))
        split_feature[0] = corrected_brain_regions
        correct_feature = " ".join(split_feature)
        corrected_cohF.append(correct_feature)
    return corrected_cohF

def augment_cohFeatures(list_of_data: list, list_of_cohFeatures: list,num_freqs=56) -> list:
    corrected_list_of_cohFeatures = [correct_cohFeatures(cohFeatures)
                                        for cohFeatures in list_of_cohFeatures]

    all_features = np.hstack(corrected_list_of_cohFeatures)
    #Get all of the unique features after naming corrections
    unique_features = np.unique(all_features)

    #Correct the ordering of frequencies
    unique_coh_features = [" ".join([feature_name.split(" ")[0],str(freq)])
                                for feature_name in unique_features[::num_freqs]
                                for freq in range(1,num_freqs+1)]

    #Store the list of unique datasets
    augmented_list_of_data = []
    augmented_list_of_masks = []

    for data,features in zip(list_of_data,corrected_list_of_cohFeatures):
        augmented_data = np.zeros((data.shape[0],len(unique_coh_features)))
        mask = np.zeros_like(augmented_data)

        for idx, feature in enumerate(features):
            f_idx = unique_coh_features.index(feature)
            augmented_data[:,f_idx] = data[:,idx]
            mask[:,f_idx] = 1

        augmented_list_of_data.append(augmented_data)
        augmented_list_of_masks.append(mask)

    return augmented_list_of_data,augmented_list_of_masks,unique_coh_features

def correct_gcFeatures(gcFeatures):
    corrected_gcF = []
    for feature in gcFeatures:
        split_feature = feature.split(' ')
        split_brain_regions = split_feature[0].split('->')

        split_brain_regions[0] = Name_Corrector_Dict[split_brain_regions[0]]
        split_brain_regions[1] = Name_Corrector_Dict[split_brain_regions[1]]

        corrected_brain_regions = "->".join(split_brain_regions)
        split_feature[0] = corrected_brain_regions
        correct_feature = " ".join(split_feature)
        corrected_gcF.append(correct_feature)
    return corrected_gcF

def augment_gcFeatures(list_of_data: list, list_of_gcFeatures: list,num_freqs=56) -> list:
    corrected_list_of_gcFeatures = [correct_gcFeatures(gcFeatures)
                                        for gcFeatures in list_of_gcFeatures]

    all_features = np.hstack(corrected_list_of_gcFeatures)
    #Get all of the unique features after naming corrections
    unique_features = np.unique(all_features)

    #Correct the ordering of frequencies
    unique_gc_features = [" ".join([feature_name.split(" ")[0],str(freq)])
                                for feature_name in unique_features[::num_freqs]
                                for freq in range(1,num_freqs+1)]

    #Store the list of unique datasets
    augmented_list_of_data = []
    augmented_list_of_masks = []

    for data,features in zip(list_of_data,corrected_list_of_gcFeatures):
        augmented_data = np.zeros((data.shape[0],len(unique_gc_features)))
        mask = np.zeros_like(augmented_data)

        for idx, feature in enumerate(features):
            f_idx = unique_gc_features.index(feature)
            augmented_data[:,f_idx] = data[:,idx]
            mask[:,f_idx] = 1

        augmented_list_of_data.append(augmented_data)
        augmented_list_of_masks.append(mask)

    return augmented_list_of_data,augmented_list_of_masks,unique_gc_features

#Returns lists of X,y, and y_mouse where each index corresponds to a fold
#Not memory efficient, but should get the job done for the moment
def lpne_k_folds_cv(X,y,y_mouse,n_folds):
    unique_mice = np.unique(y_mouse)
    kf = KFold(n_splits=n_folds)
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    y_mouse_train_list = []
    y_mouse_test_list = []

    for train_mice_idxs, test_mice_idxs in kf.split(unique_mice):
        train_mice = unique_mice[train_mice_idxs]
        test_mice = unique_mice[test_mice_idxs]
        train_mask = np.zeros(len(y_mouse)).astype(bool)
        test_mask = np.zeros(len(y_mouse)).astype(bool)
        
        for mouse in train_mice:
            current_mouse_mask = (np.array(y_mouse) == mouse)
            train_mask = np.logical_or(train_mask,current_mouse_mask)
        
        for mouse in test_mice:
            current_mouse_mask = (np.array(y_mouse) == mouse)
            test_mask = np.logical_or(test_mask,current_mouse_mask)
        
        X_train_fold = X[train_mask==1]
        X_test_fold = X[test_mask==1]
        y_train_fold = y[train_mask==1]
        y_test_fold = y[test_mask==1]
        mouse_train_fold = y_mouse[train_mask==1]
        mouse_test_fold = y_mouse[test_mask==1]

        X_train_list.append(X_train_fold)
        X_test_list.append(X_test_fold)
        y_train_list.append(y_train_fold)
        y_test_list.append(y_test_fold)
        y_mouse_train_list.append(mouse_train_fold)
        y_mouse_test_list.append(mouse_test_fold)

    return X_train_list, X_test_list, y_train_list, y_test_list, y_mouse_train_list, y_mouse_test_list

#Does train test split by mouse according to y_mouse. This helps seperate patients (mice)
def lpne_train_test_split(X,y,y_mouse,test_size=0.2,validation=False,random_state=None):
    UNIQUE_MICE = np.unique(y_mouse)

    train_mice, test_mice = train_test_split(UNIQUE_MICE,test_size=test_size,random_state=random_state)

    train_mouse_idxs_dict = {}
    test_mouse_idxs_dict = {}

    for mouse in train_mice:
        train_mouse_idxs_dict[mouse] = np.where(y_mouse==mouse)[0]

    for mouse in test_mice:
        test_mouse_idxs_dict[mouse] = np.where(y_mouse==mouse)[0]

    X_train = np.vstack([X[train_mouse_idxs_dict[mouse],:] for mouse in train_mice])
    y_train = np.hstack([y[train_mouse_idxs_dict[mouse]] for mouse in train_mice])
    y_train_mouse = np.hstack([y_mouse[train_mouse_idxs_dict[mouse]] for mouse in train_mice])

    X_test = np.vstack([X[test_mouse_idxs_dict[mouse],:] for mouse in test_mice])
    y_test = np.hstack([y[test_mouse_idxs_dict[mouse]] for mouse in test_mice])
    y_test_mouse = np.hstack([y_mouse[test_mouse_idxs_dict[mouse]] for mouse in test_mice])

    return X_train, X_test, y_train, y_test, y_train_mouse, y_test_mouse

#Collects auc per mouse and stores it in a dictionary
#Supports using mannWhitneyU to get auc
def lpne_auc(y_pred,y_true,y_mouse,z=None,mannWhitneyU=False):
    auc_Dict = {}
    if mannWhitneyU:
        auc_Dict['auc_method']='mannWhitneyU'
        assert z is not None
        for mouse in np.unique(y_mouse):
            mouse_mask = y_mouse==mouse
            z_pos_mouse = z[np.logical_and(mouse_mask==1,y_true==1)==1,0]
            z_neg_mouse = z[np.logical_and(mouse_mask==1,y_true==0)==1,0]

            if z_pos_mouse.shape[0]==0 or z_neg_mouse.shape[0]==0:
                print("Mouse ",mouse, " has only one class - AUC cannot be calculated")
                print("n_positive samples ",z_pos_mouse.shape[0])
                print("n_negative samples ",z_neg_mouse.shape[0])
                auc_Dict[mouse] = (np.nan,np.nan)
            else:
                U,pval = mannwhitneyu(z_pos_mouse,z_neg_mouse)
                mw_auc = U/(len(z_pos_mouse)*len(z_neg_mouse))
                auc_Dict[mouse] = (mw_auc,pval)
    else:
        auc_Dict['auc_method']='sklearn_roc_auc'
        for mouse in np.unique(y_mouse):
            mouse_mask = y_mouse==mouse

            try:
                auc_Dict[mouse] = roc_auc_score(y_true[mouse_mask==1],y_pred[mouse_mask==1])
            except:
                fpr,tpr,thresh = roc_curve(y_true[mouse_mask==1],y_pred[mouse_mask==1])
                auc_Dict[mouse] = auc(fpr,tpr)

    return auc_Dict

def get_mean_std_err_auc(y_pred,y_true,y_mouse,z=None,mannWhitneyU=False):
    auc_dict = lpne_auc(y_pred,y_true,y_mouse,z,mannWhitneyU)
    if mannWhitneyU:
        auc_list = [auc_dict[mouse][0] for mouse in auc_dict.keys() - ['auc_method']]# if auc_dict[mouse][0] == auc_dict[mouse][0]]
        mean = np.mean(auc_list)
        std = np.std(auc_list) / np.sqrt(len(auc_list)-1)
    else:
        auc_list = [auc_dict[mouse] for mouse in auc_dict.keys() - ['auc_method']]# if auc_dict[mouse] == auc_dict[mouse]]
        mean = np.mean(auc_list)
        std = np.std(auc_list) / np.sqrt(len(auc_list)-1)
    return mean, std

def make_projection_csv(pickle_file,model,X_feature_list,other_features,save_file,auc_dict=None,auc_type="mw",weights=None):

    with open(pickle_file,'rb') as f:
        project_dict = pickle.load(f)
    if weights is None:
        X_project = np.hstack([project_dict[feature] for feature in X_feature_list])
    else:
        X_project = np.hstack([project_dict[feature]*weight for feature,weight in zip(X_feature_list,weights)])
    s = model.project(X_project)
    s = s[:,0]
    save_dict = {}
    save_dict['scores'] = s

    for feature in other_features:
        save_dict[feature] = project_dict[feature]

    if auc_dict is not None:
        y_mouse_array = np.array(project_dict['y_mouse'])
        auc_array = np.ones(len(y_mouse_array))*np.nan
        if auc_type=="mw":
            p_val_array = np.ones(len(y_mouse_array))*np.nan
        for mouse in auc_dict.keys() - ['auc_method']:

            mask = y_mouse_array==mouse
            if auc_type=="mw":
                auc_array[mask==1] = auc_dict[mouse][0]
                p_val_array[mask==1]= auc_dict[mouse][1]
            else:
                auc_array[mask==1]=auc_dict[mouse]

        save_dict['auc'] = auc_array
        if auc_type=="mw": save_dict['p_val'] = p_val_array

    df = pd.DataFrame.from_dict(save_dict)
    df.to_csv(save_file,index=False,header=True)

    return df

def make_recon_plots(model,X,sample,task=" ",title=None,skl_mse=None,nn_mse=None,saveFile=None):

    if skl_mse is None and nn_mse is None:
        skl_mse, nn_mse = model.get_skl_nn_mse(X)
    X_recon,X_recon_z1,y_pred_proba,s = model.transform(X)
    X_recon_z1 = model.get_comp_recon(X,0)
    plt.figure(figsize=(20,8))
    plt.subplot(1,2,1)
    plt.plot(X_recon[sample,:],alpha=0.5,label='Recon')
    plt.plot(X_recon_z1[sample,:],alpha=0.5,label='Sup Comp Recon')
    plt.plot(X[sample,:],alpha=0.5,label='Original Signal')
    plt.title("{} data Window {}, Unsup MSE: {:2.4f}, dCSFA MSE: {:2.4f}".format(task,sample,skl_mse,nn_mse))
    plt.xlabel("Feature (Every 56 indices is a feature group)")
    plt.ylabel("Magnitude")
    plt.legend()


    plt.subplot(1,2,2)
    plt.plot(X_recon[sample,:],alpha=0.5,label='Recon')
    plt.plot(X_recon_z1[sample,:],alpha=0.5,label='Sup Comp Recon')
    plt.title("{} data first component reconstruction".format(task))
    plt.xlabel("Feature (Every 56 indices is a feature group)")
    plt.ylabel("Magnitude")
    plt.legend()
    if saveFile is not None:
        plt.savefig(saveFile)
    plt.show()

def getReconContribution(X,n_components,model):
    perc_contribution_list = []
    EPSILON = 1e-6
    X_recon,_,_,s = model.transform(X)
    for component in range(n_components):
        X_recon_comp = model.get_comp_recon(s,component)
        perc_contribution = np.divide(X_recon_comp+EPSILON,X_recon+EPSILON)
        avg_perc_contribution = np.mean(perc_contribution,axis=0)

        perc_contribution_list.append(avg_perc_contribution)
    
    perc_contribution_mat = np.vstack(perc_contribution_list)
    return perc_contribution_mat


def makeUpperTriangularPlot_pow_coh_gc(X,areas,psdFeatures,cohFeatures,gcFeatures,
                                        freq=56,net_idx=0,saveFile='demo.png',
                                        title="Supervised Network Percent Recon Contribution",
                                        figsize=(30,20), silenceTicks=True):
    
    plt.figure(figsize=figsize)
    num_areas = len(areas)
    X_psd = X[:,:len(psdFeatures)]
    X_coh = X[:,len(psdFeatures):(len(psdFeatures)+len(cohFeatures))]
    X_gc = X[:,(len(psdFeatures)+len(cohFeatures)):]

    print(X_psd.shape,X_coh.shape,X_gc.shape)
    for idx,area in enumerate(areas):
        plt.subplot(num_areas,num_areas,idx+1 + num_areas*idx)
        plt.plot(range(1,freq+1),X_psd[net_idx,idx*56:(idx+1)*56])
        plt.ylabel(area)
        plt.xlabel("Freq")
        plt.ylim([0,1])

        if idx == 0 or idx == num_areas-1:
            plt.title(area)

    #coherence features
    reshape_coherence_features = cohFeatures.reshape(-1,56)
    for feature_idx in range(reshape_coherence_features.shape[0]):
        feature = reshape_coherence_features[feature_idx,0]
        area_1, left_over = feature.split('-')
        area_2, _ = left_over.split(' ')

        idx_area_1 = areas.index(area_1)
        idx_area_2 = areas.index(area_2)

        subplot_idx = idx_area_1*num_areas + idx_area_2 + 1


        plt.subplot(num_areas,num_areas,subplot_idx)
        #print(subplot_idx,X_coh[net_idx,feature_idx*freq:(feature_idx+1)*freq])
        plt.plot(range(1,freq+1),X_coh[net_idx,feature_idx*freq:(feature_idx+1)*freq])

        if silenceTicks:
            plt.yticks([])
            plt.xticks([])
        plt.ylim([0,1])
    reshape_gc_features = gcFeatures.reshape(-1,56)
    for feature_idx in range(reshape_gc_features.shape[0]):
        feature = reshape_gc_features[feature_idx,0]
        area_1, left_over = feature.split('->')
        area_2, _ = left_over.split(' ')

        idx_area_1 = areas.index(area_1)
        idx_area_2 = areas.index(area_2)

        if idx_area_1 < idx_area_2:
            subplot_idx = idx_area_1*num_areas + idx_area_2 + 1

            plt.subplot(num_areas,num_areas,subplot_idx)
            plt.plot(range(1,freq+1),X_gc[0,feature_idx*freq:(feature_idx+1)*freq]/2 +0.5,color="green")
            plt.axhline(0.5,alpha=0.5,color='gray')
            plt.ylim([0,1])

            if silenceTicks:
                plt.yticks([])
                plt.xticks([])
            if subplot_idx%(num_areas)==0:
                ax2 = plt.twinx()
                ax2.set_ylim(-1,1)
            
            if subplot_idx < num_areas:
                plt.title(areas[subplot_idx-1])
        else:
            #Flip the index order to stay upper triangular
            subplot_idx = idx_area_2*num_areas + idx_area_1 + 1
            plt.subplot(num_areas,num_areas,subplot_idx)
            plt.plot(range(1,freq+1),-X_gc[net_idx,feature_idx*freq:(feature_idx+1)*freq]/2 + 0.5,color="red")
            plt.axhline(0.5,alpha=0.5,color='gray')
            plt.ylim([0,1])

            if silenceTicks:
                plt.yticks([])
                plt.xticks([])
            if subplot_idx%(num_areas)==0:
                ax2 = plt.twinx()
                ax2.set_ylim(-1,1)
            
            if subplot_idx < num_areas:
                plt.title(areas[subplot_idx-1])
                
    plt.suptitle(title)
    plt.savefig(saveFile)
    plt.show()


def makeUpperTriangularPlot_pow_ds(X,areas,psdFeatures,dsFeatures,
                                        freq=56,net_idx=0,saveFile='demo.png',
                                        title="Supervised Network Percent Recon Contribution",
                                        figsize=(30,20), silenceTicks=True):
    
    plt.figure(figsize=figsize)
    num_areas = len(areas)
    X_psd = X[:,:len(psdFeatures)]
    X_ds = X[:,len(psdFeatures):]

    print(X_psd.shape,X_ds.shape)
    for idx,area in enumerate(areas):
        plt.subplot(num_areas,num_areas,idx+1 + num_areas*idx)
        plt.plot(range(1,freq+1),X_psd[net_idx,idx*56:(idx+1)*56])
        plt.ylabel(area)
        plt.xlabel("Freq")
        plt.ylim([0,1])

        if idx == 0 or idx == num_areas-1:
            plt.title(area)

    reshape_ds_features = dsFeatures.reshape(-1,56)
    for feature_idx in range(reshape_ds_features.shape[0]):
        feature = reshape_ds_features[feature_idx,0]
        area_1, left_over = feature.split('->')
        area_2, _ = left_over.split(' ')

        idx_area_1 = areas.index(area_1)
        idx_area_2 = areas.index(area_2)

        if idx_area_1 < idx_area_2:
            subplot_idx = idx_area_1*num_areas + idx_area_2 + 1

            plt.subplot(num_areas,num_areas,subplot_idx)
            plt.plot(range(1,freq+1),X_ds[0,feature_idx*freq:(feature_idx+1)*freq]/2 +0.5,color="green")
            plt.axhline(0.5,alpha=0.5,color='gray')
            plt.ylim([0,1])

            if silenceTicks:
                plt.yticks([])
                plt.xticks([])
            if subplot_idx%(num_areas)==0:
                ax2 = plt.twinx()
                ax2.set_ylim(-1,1)
            
            if subplot_idx < num_areas:
                plt.title(areas[subplot_idx-1])
        else:
            #Flip the index order to stay upper triangular
            subplot_idx = idx_area_2*num_areas + idx_area_1 + 1
            plt.subplot(num_areas,num_areas,subplot_idx)
            plt.plot(range(1,freq+1),-X_ds[net_idx,feature_idx*freq:(feature_idx+1)*freq]/2 + 0.5,color="red")
            plt.axhline(0.5,alpha=0.5,color='gray')
            plt.ylim([0,1])

            if silenceTicks:
                plt.yticks([])
                plt.xticks([])
            if subplot_idx%(num_areas)==0:
                ax2 = plt.twinx()
                ax2.set_ylim(-1,1)
            
            if subplot_idx < num_areas:
                plt.title(areas[subplot_idx-1])
                
    plt.suptitle(title)
    plt.savefig(saveFile)
    plt.show()
'''class data_Augmenter(object):
    def __init__(self,list_of_label_lists,imp_strat='zero'):
        super(data_Augmenter,self).__init__()
        self.unique_labels_list = unique_labels_list
        self.imp_strat = imp_strat

    def _get_unique_labels(self,list_of_label_lists):
        master_list = []
        for label_list in list_of_label_lists:
            master_list += label_list
        unique_labels = list(np.unique(master_list))
        return unique_labels

    def augment(self,X,label_list):
        assert X.shape[1] == len(label_list)

        n_samps = X.shape[0]
        X_aug = np.zeros((n_samps,len(self.unique_labels_list)))
        M_aug = np.zeros_like(X_aug)

        for loc_idx,feature in enumerate(label_list):
            global_idx = self.unique_labels_list.index(feature)

            X_aug[:,global_idx] = X[:,local_idx]
            M_aug[:,global_idx] = 1

        return X_aug, M_aug'''
