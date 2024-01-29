import pandas as pd
import numpy as np

import os
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


def data_download(dataset):
    """
    Load dataset
    :param dataset: name of the dataset target combination
    :return: dataset, target
    """
    # Experiment Datasets
    ###########################################
    if dataset == "dataset_diabetes":
        dataset_name = "dataset_diabetes"
        target = "readmitted_yesno"
    elif dataset == "dataset_diabetesNA60":
        dataset_name = "dataset_diabetesNA60"
        target = "readmitted_yesnoNA60"
    elif dataset == "dataset_diabetesNA60_small":
        dataset_name = "dataset_diabetesNA60_small"
        target = "readmitted_yesnoNA60"
    elif dataset == "dataset_diabetes_small":
        dataset_name = "dataset_diabetes_small"
        target = "readmitted_yesno"
    elif dataset == "dataset_diabetes_tiny":
        dataset_name = "dataset_diabetes_tiny"
        target = "readmitted_yesno"
    elif dataset == "dataset_diabetesNA60_tiny":
        dataset_name = "dataset_diabetesNA60_tiny"
        target = "readmitted_yesnoNA60"
    elif dataset == "dataset_diabetesNA60_small_tiny":
        dataset_name = "dataset_diabetesNA60_small_tiny"
        target = "readmitted_yesnoNA60"
    elif dataset == "dataset_diabetes_small_tiny":
        dataset_name = "dataset_diabetes_small_tiny"
        target = "readmitted_yesno"
    elif dataset == "dataset_myocardial_mortality":
        dataset_name = "dataset_myocardial_mortality"
        target = "mortality"
    elif dataset == "dataset_myocardial_mortality_NA60":
        dataset_name = "dataset_myocardial_mortality_NA60"
        target = "mortalityNA60"
    elif dataset == "dataset_myocardial_mortality_NA60_small":
        dataset_name = "dataset_myocardial_mortality_NA60_small"
        target = "mortalityNA60"
    elif dataset == "dataset_myocardial_mortality_small":
        dataset_name = "dataset_myocardial_mortality_small"
        target = "mortality"
    elif dataset == "dataset_myocardial_REC_IM":
        dataset_name = "dataset_myocardial_REC_IM"
        target = "REC_IM122"
    elif dataset == "dataset_myocardial_REC_IM_NA60":
        dataset_name = "dataset_myocardial_REC_IM_NA60"
        target = "REC_IM122NA60"
    elif dataset == "dataset_myocardial_REC_IM_NA60_small":
        dataset_name = "dataset_myocardial_REC_IM_NA60_small"
        target = "REC_IM122NA60"
    elif dataset == "dataset_myocardial_REC_IM_small":
        dataset_name = "dataset_myocardial_REC_IM_small"
        target = "REC_IM122"
    elif dataset == "dataset_thyroid":
        dataset_name = "dataset_thyroid"
        target = "21"
    elif dataset == "dataset_thyroidNA60":
        dataset_name = "dataset_thyroidNA60"
        target = "targetNA60"
    elif dataset == "dataset_thyroid_small":
        dataset_name = "dataset_thyroid_small"
        target = "21"
    elif dataset == "dataset_thyroidNA60_small":
        dataset_name = "dataset_thyroidNA60_small"
        target = "targetNA60"
    elif dataset == "dataset_thyroidNA20":
        dataset_name = "dataset_thyroidNA20"
        target = "21"
    elif dataset == "dataset_thyroidNA20_small":
        dataset_name = "dataset_thyroidNA20_small"
        target = "21"
    elif dataset == "dataset_thyroidNA20NA60":
        dataset_name = "dataset_thyroidNA20NA60"
        target = "targetNA60"
    elif dataset == "dataset_thyroidNA20NA60_small":
        dataset_name = "dataset_thyroidNA20NA60_small"
        target = "targetNA60"
    elif dataset == "dataset_thyroidNA10":
        dataset_name = "dataset_thyroidNA10"
        target = "21"
    elif dataset == "dataset_thyroidNA10_small":
        dataset_name = "dataset_thyroidNA10_small"
        target = "21"
    elif dataset == "dataset_thyroidNA10NA60":
        dataset_name = "dataset_thyroidNA10NA60"
        target = "targetNA60"
    elif dataset == "dataset_thyroidNA10NA60_small":
        dataset_name = "dataset_thyroidNA10NA60_small"
        target = "targetNA60"
    elif dataset == "dataset_thyroidNA30":
        dataset_name = "dataset_thyroidNA30"
        target = "21"
    elif dataset == "dataset_thyroidNA30_small":
        dataset_name = "dataset_thyroidNA30_small"
        target = "21"
    elif dataset == "dataset_thyroidNA30NA60":
        dataset_name = "dataset_thyroidNA30NA60"
        target = "targetNA60"
    elif dataset == "dataset_thyroidNA30NA60_small":
        dataset_name = "dataset_thyroidNA30NA60_small"
        target = "targetNA60"

        ###########################################
    else:
        print('TODO: HAVE TO DO THIS DATASET!')

    # Experiment Datasets
    ###################################################
    if dataset == "dataset_diabetes" or dataset == "dataset_diabetesNA60":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","diabetes.csv"), low_memory=False)
        out_cols = ["readmitted_yesno","readmitted_yesnoNA60"]
        out_cols = [col for col in out_cols if not col== target]
        cols = [col for col in train.columns if not col in out_cols]
        train = train[cols]
    elif dataset == "dataset_diabetes_tiny" or dataset == "dataset_diabetesNA60_tiny":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","diabetes_tiny.csv"), low_memory=False)
        out_cols = ["readmitted_yesno","readmitted_yesnoNA60"]
        out_cols = [col for col in out_cols if not col== target]
        cols = [col for col in train.columns if not col in out_cols]
        train = train[cols]
    elif dataset == "dataset_diabetes_small":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","diabetes.csv"), low_memory=False)
        cols = ['discharge_disposition_id', 'number_inpatient', 'admission_source_id',
                'diag_1', 'diag_2', 'medical_specialty', 'payer_code',
                'number_emergency', 'admission_type_id', 'diabetesMed', 'age',
                'number_outpatient', 'diag_3', 'num_lab_procedures',
                'time_in_hospital', 'weight', 'number_diagnoses', 'num_medications',
                'num_procedures', 'insulin', "Set", "readmitted_yesno"]
        train = train[cols]
    elif dataset == "dataset_diabetes_small_tiny":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","diabetes_tiny.csv"), low_memory=False)
        cols = ['discharge_disposition_id', 'number_inpatient', 'diag_3', 'diabetesMed', 'num_lab_procedures', 'number_diagnoses', 'num_procedures', 'payer_code', 'number_emergency', 'insulin', 'admission_source_id', 'race', 'medical_specialty', 'admission_type_id', 'gender', 'change', 'A1Cresult', 'weight', 'time_in_hospital', 'number_outpatient', 'max_glu_serum', 'nateglinide', "Set", "readmitted_yesno"]
        train = train[cols]
    elif dataset == "dataset_diabetesNA60_small":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","diabetes.csv"), low_memory=False)
        cols = ['discharge_disposition_id', 'number_inpatient',
                'admission_source_id', 'number_emergency', 'diag_1', 'payer_code',
                'weight', 'medical_specialty', 'number_diagnoses',
                'admission_type_id', 'number_outpatient', 'diag_2',
                'num_medications', 'num_procedures', 'time_in_hospital',
                'diabetesMed', 'age', 'insulin', 'A1Cresult',
                'nateglinide', 'tolazamide', "Set", "readmitted_yesnoNA60"]
        train = train[cols]
    elif dataset == "dataset_diabetesNA60_small_tiny":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","diabetes_tiny.csv"), low_memory=False)
        cols = ['diag_3', 'admission_type_id', 'number_diagnoses', 'discharge_disposition_id', 'number_inpatient', 'A1Cresult', 'change', 'diabetesMed', 'gender', 'rosiglitazone', 'acarbose', 'glyburide-metformin', "Set", "readmitted_yesnoNA60"]
        train = train[cols]
    elif dataset == "dataset_myocardial_mortality" or dataset == "dataset_myocardial_mortality_NA60":
        train = pd.read_csv(os.path.join(os.getcwd(), "datasets", "myocardial.csv"), low_memory=False)
        out_cols = ["FIBR_PREDS113","PREDS_TAH114","JELUD_TAH115","FIBR_JELUD116","A_V_BLOK117","OTEK_LANC118",
                    "RAZRIV119","DRESSLER120","ZSN121","REC_IM122","P_IM_STEN123","LET_IS124","mortality",
                    "mortalityNA60","REC_IM122NA60"]
        out_cols = [col for col in out_cols if not col == target]
        cols = [col for col in train.columns if not col in out_cols]
        train = train[cols]
    elif dataset == "dataset_myocardial_mortality_small":
        train = pd.read_csv(os.path.join(os.getcwd(), "datasets", "myocardial.csv"), low_memory=False)
        cols = ['ZSN_A12', 'n_p_ecg_p_12_75', 'K_SH_POST40', 'inf_im47', 'TIME_B_S92',
                'zab_leg_02_31', 'IBS_POST7', 'age2', 'ROE91', 'endocr_02_28', 'STENOK_AN5',
                'NA_KB96', 'n_p_ecg_p_03_66', 'D_AD_ORIT38', 'L_BLOOD90', 'MP_TP_POST41',
                'ritm_ecg_p_01_50', 'S_AD_ORIT37', 'S_AD_KBRIG35', 'Na_BLOOD86', 'K_BLOOD84',
                'endocr_01_27', 'ritm_ecg_p_07_54', 'ALT_BLOOD87', 'AST_BLOOD88', 'nr04_17',
                'IM_PG_P49', 'ritm_ecg_p_02_51', 'LID_S_n106', 'post_im48', 'n_p_ecg_p_07_70',
                'ritm_ecg_p_08_55', 'ritm_ecg_p_04_52', 'n_r_ecg_p_01_56', 'n_p_ecg_p_06_69',"Set","mortality"]
        train = train[cols]
    elif dataset == "dataset_myocardial_mortality_NA60_small":
        train = pd.read_csv(os.path.join(os.getcwd(), "datasets", "myocardial.csv"), low_memory=False)
        cols = ['ZSN_A12', 'inf_im47', 'ANT_CA_S_n108', 'zab_leg_02_31', 'age2', 'lat_im46',
                'K_SH_POST40', 'S_AD_KBRIG35', 'IBS_POST7', 'NA_KB96', 'STENOK_AN5',
                'n_p_ecg_p_12_75', 'AST_BLOOD88', 'post_im48', 'LID_KB98', 'endocr_01_27',
                'ritm_ecg_p_02_51', 'O_L_POST39', 'nr04_17', 'ritm_ecg_p_04_52',
                'n_p_ecg_p_06_69', 'S_AD_ORIT37', 'ritm_ecg_p_01_50', 'ritm_ecg_p_07_54',
                'INF_ANAM4', 'GEPAR_S_n109', 'n_r_ecg_p_05_60', 'SIM_GIPERT10', 'nr03_16',
                'gender3', 'nr02_15', 'nr11_13',"Set","mortalityNA60"]
        train = train[cols]
    elif dataset == "dataset_myocardial_REC_IM" or dataset == "dataset_myocardial_REC_IM_NA60":
        train = pd.read_csv(os.path.join(os.getcwd(), "datasets", "myocardial.csv"), low_memory=False)
        out_cols = ["FIBR_PREDS113","PREDS_TAH114","JELUD_TAH115","FIBR_JELUD116","A_V_BLOK117","OTEK_LANC118",
                    "RAZRIV119","DRESSLER120","ZSN121","REC_IM122","P_IM_STEN123","LET_IS124","mortality",
                    "mortalityNA60","REC_IM122NA60"]
        out_cols = [col for col in out_cols if not col == target]
        cols = [col for col in train.columns if not col in out_cols]
        train = train[cols]
    elif dataset == "dataset_myocardial_REC_IM_small":
        train = pd.read_csv(os.path.join(os.getcwd(), "datasets", "myocardial.csv"), low_memory=False)
        cols = ['STENOK_AN5', 'L_BLOOD90', 'TIME_B_S92', 'gender3', 'Na_BLOOD86', 'S_AD_KBRIG35',
               'GEPAR_S_n109', 'n_p_ecg_p_07_70', 'AST_BLOOD88', 'NA_KB96', 'ritm_ecg_p_01_50',
               'LID_KB98', 'ASP_S_n110', 'LID_S_n106', 'n_r_ecg_p_03_58', 'ant_im45',
               'endocr_01_27', 'n_r_ecg_p_06_61', 'n_r_ecg_p_04_59', 'IBS_POST7', 'IBS_NASL8',
               'ritm_ecg_p_02_51', 'B_BLOK_S_n107', 'ritm_ecg_p_04_52',"Set",'REC_IM122']
        train = train[cols]
    elif dataset == "dataset_myocardial_REC_IM_NA60_small":
        train = pd.read_csv(os.path.join(os.getcwd(), "datasets", "myocardial.csv"), low_memory=False)
        cols = ['gender3', 'L_BLOOD90', 'age2', 'NA_KB96', 'S_AD_ORIT37', 'TIME_B_S92',
                'endocr_01_27', 'LID_KB98', 'NITR_S99', 'zab_leg_01_30', 'n_p_ecg_p_11_74',
                'IBS_POST7', 'LID_S_n106', 'ZSN_A12', 'n_p_ecg_p_07_70', 'B_BLOK_S_n107',
                'zab_leg_02_31', 'inf_im47', 'nr03_16', 'n_p_ecg_p_06_69', 'n_p_ecg_p_03_66',
                'n_p_ecg_p_12_75', 'TIKL_S_n111', 'SIM_GIPERT10',"Set",'REC_IM122NA60']
        train = train[cols]
    elif dataset == "dataset_thyroid" or dataset=="dataset_thyroidNA60":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","thyroid.csv"),
            low_memory=False)
        out_cols = ["16","21","targetNA60"]
        cols = [col for col in train.columns if not col in out_cols]
        cols = [*cols,target]
        train=train[cols]
    elif dataset == "dataset_thyroidNA20" or dataset=="dataset_thyroidNA20NA60":
        train = pd.read_csv(os.path.join(os.getcwd(), "datasets", "thyroidNA20.csv"), low_memory=False)
        out_cols = ["16","21","targetNA60"]
        cols = [col for col in train.columns if not col in out_cols]
        cols = [*cols, target]
        train = train[cols]
    elif dataset == "dataset_thyroidNA10" or dataset=="dataset_thyroidNA10NA60":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","thyroidNA10.csv"), low_memory=False)
        out_cols = ["16","21","targetNA60"]
        cols = [col for col in train.columns if not col in out_cols]
        cols = [*cols, target]
        train = train[cols]
    elif dataset == "dataset_thyroidNA30" or dataset=="dataset_thyroidNA30NA60":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","thyroidNA30.csv"), low_memory=False)
        out_cols = ["16","21","targetNA60"]
        cols = [col for col in train.columns if not col in out_cols]
        cols = [*cols, target]
        train = train[cols]
    elif dataset == "dataset_thyroid_small":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","thyroid.csv"),
            low_memory=False)
        cols = ['17', '20', '19', '18', '2', '1', '0', '7', '13', '6', '12', '3',"Set",'21']
        train = train[cols]
    elif dataset == "dataset_thyroidNA60_small":
        train = pd.read_csv(os.path.join(os.getcwd(), "datasets", "thyroid.csv"), low_memory=False)
        cols = ['17', '20', '19', '1', '0', '9', '2', '6',"Set","targetNA60"]
        train = train[cols]
    elif dataset == "dataset_thyroidNA20_small":
        train = pd.read_csv(os.path.join(os.getcwd(), "datasets", "thyroidNA20.csv"), low_memory=False)
        cols = ['20', '17', '18', '19', '2', '0', '1', '8', '9',"Set",'21']
        train = train[cols]
    elif dataset == "dataset_thyroidNA20NA60_small":
        train = pd.read_csv(os.path.join(os.getcwd(), "datasets", "thyroidNA20.csv"), low_memory=False)
        cols = ['17', '19', '20', '18', '9', '1', '2', '0', '6', '5', '8', '4',"Set","targetNA60"]
        train = train[cols]

    elif dataset == "dataset_thyroidNA10_small":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","thyroidNA10.csv"), low_memory=False)
        cols = ['20', '17', '18', '2', '19', '1', '0', '11', '7', '6', '4',"Set",'21']
        train = train[cols]
    elif dataset == "dataset_thyroidNA10NA60_small":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","thyroidNA10.csv"), low_memory=False)
        cols = ['20', '17', '19', '0', '18', '1', '3', '9', '12', '13', '4',"Set","targetNA60"]
        train = train[cols]

    elif dataset == "dataset_thyroidNA30_small":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","thyroidNA30.csv"), low_memory=False)
        cols = ['18', '17', '20', '19', '0', '2', '1', '9', '10', '8',"Set",'21']
        train = train[cols]
    elif dataset == "dataset_thyroidNA30NA60_small":
        train = pd.read_csv(os.path.join(os.getcwd(),"datasets","thyroidNA30.csv"), low_memory=False)
        cols = ['17', '18', '19', '20', '2', '7', '5', '1',"Set","targetNA60"]
        train = train[cols]
    else:
        print("No Dataset is selected")
        train = None
    ###################################################

    return train, target

        
def data_mask_split(X,y,mask,y_mask,indices,mask_det,stage):
    try:
        x_d = {
            'data': X.values[indices],
            'mask': mask.values[indices]
        }
    except:
        x_d = {
            'data': X.values[indices],
            'mask': mask[indices]
        }
    y_d = {
        'data': y.values[indices].reshape(-1, 1),
        'mask': y_mask[indices].reshape(-1, 1)
    } 
    if mask_det is not None:
        if stage == 'train' and mask_det['avail_train_y'] > 0:
            avail_ys = np.random.choice(y_d['mask'].shape[0], mask_det['avail_train_y'], replace=False)
            y_d['mask'][avail_ys,:] = 1
        
        if stage != 'train' and mask_det['test_mask'] < 10e-3:
            x_d['mask'] = np.ones_like(x_d['mask'])

    return x_d, y_d


def data_prep(dataset,seed):
    """
    Prepare dataset for training
    All categorical features will be numerical encoded and missing values will be imputed by a new category
    All continuous features will be standardized and missing values will be imputed by the mean
    :param dataset: dataset
    :param seed: seed
    :return: cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std

    cat_dims: dimension of each categorical feature
    cat_idxs: index of each categorical feature
    con_idxs: index of each continuous feature
    X_train: training dataset only with features
    y_train: training dataset only with target
    X_valid: validation dataset only with features
    y_valid: validation dataset only with target
    X_test: test dataset only with features
    y_test: test dataset only with target
    train_mean: mean of each continuous feature on the training dataset
    train_std: standard deviation of each continuous feature on the training dataset
    """
    np.random.seed(seed)

    # Experiment Dataset Preprocessing
    #########################################################
    train, target = data_download(dataset)
    unused_feat = ['Set']
    features = [col for col in train.columns if col not in unused_feat + [target]]
    train["CLS"] = "CLS"
    features = ["CLS", *features]
    train = train[[*features, *unused_feat, target]]
    temp = train.fillna("ThisisNan")
    train_indices = train[train.Set == "train"].index
    valid_indices = train[train.Set == "valid"].index
    test_indices = train[train.Set == "test"].index
    categorical_columns = []
    categorical_dims = {}
    for col in train.columns[train.dtypes == object]:
        if not col == target:
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("ZZZ_likely")
            train[col] = l_enc.fit_transform(train[col].values)

    for col in train.columns[train.dtypes == 'float64']:
        train[col].fillna(train.loc[train_indices, col].mean(), inplace=True)
    for col in train.columns[train.dtypes == 'int64']:
        train[col].fillna(train.loc[train_indices, col].mean(), inplace=True)

    cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
    con_idxs = list(set(range(len(features))) - set(cat_idxs))
    cat_dims = [categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
    train[target] = train[target].astype(int)
    mask_det = None
    #########################################################

    # mask and y_mask to know which values are NA
    #############################################
    temp1 = temp[features]
    temp2 = temp[target]
    mask = temp1.ne("ThisisNan").astype(int)
    y_mask = np.array(temp2.ne("ThisisNan").astype(int)).reshape((-1,1))
    #############################################

    X = train[features]
    Y = train[target]
    X_train, y_train = data_mask_split(X,Y,mask,y_mask,train_indices,mask_det,'train')
    X_valid, y_valid = data_mask_split(X,Y,mask,y_mask,valid_indices,mask_det,'valid')
    X_test, y_test = data_mask_split(X,Y,mask,y_mask,test_indices,mask_det,'test')
    
    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols,continuous_mean_std=None, is_pretraining=False,tag=None):
        cat_cols = list(cat_cols)
        X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.is_pretraining = is_pretraining
        self.tag = tag
        self.X1 = X[:,cat_cols].copy().astype(np.int64) # categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) # numerical columns

        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std
            self.y = Y['data']
            self.y_mask = Y['mask']
            
        else:
            self.y = np.expand_dims(np.array(Y['data']),axis=-1)
            self.y_mask = np.expand_dims(np.array(Y['mask']),axis=-1)
        
        self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns

        # Only use labled samples for finetuning
        #######################################
        if not self.is_pretraining:
            self.X1 = self.X1[np.array(self.y_mask,dtype=bool)[:,0],:]
            self.X2 = self.X2[np.array(self.y_mask,dtype=bool)[:,0],:]
            self.X1_mask = self.X1_mask[np.array(self.y_mask,dtype=bool)[:,0],:]
            self.X2_mask = self.X2_mask[np.array(self.y_mask, dtype=bool)[:, 0], :]
            self.y = self.y[np.array(self.y_mask,dtype=bool)[:,0],:]
        #######################################

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        if self.is_pretraining:
            return self.X1[idx], self.X2[idx], self.X1_mask[idx], self.X2_mask[idx]

        else:
            return self.X1[idx], self.X2[idx], self.y[idx]