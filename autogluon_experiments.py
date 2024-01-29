from autogluon.tabular import TabularPredictor
import pandas as pd
import os


# important feature sets
important_feature_list={"myocardial":{"mortalityNA60":['ZSN_A12', 'inf_im47', 'ANT_CA_S_n108', 'zab_leg_02_31', 'age2', 'lat_im46',
                                         'K_SH_POST40', 'S_AD_KBRIG35', 'IBS_POST7', 'NA_KB96', 'STENOK_AN5',
                                         'n_p_ecg_p_12_75', 'AST_BLOOD88', 'post_im48', 'LID_KB98', 'endocr_01_27',
                                         'ritm_ecg_p_02_51', 'O_L_POST39', 'nr04_17', 'ritm_ecg_p_04_52',
                                         'n_p_ecg_p_06_69', 'S_AD_ORIT37', 'ritm_ecg_p_01_50', 'ritm_ecg_p_07_54',
                                         'INF_ANAM4', 'GEPAR_S_n109', 'n_r_ecg_p_05_60', 'SIM_GIPERT10', 'nr03_16',
                                         'gender3', 'nr02_15', 'nr11_13'],
                            "mortality":['ZSN_A12', 'n_p_ecg_p_12_75', 'K_SH_POST40', 'inf_im47', 'TIME_B_S92',
                                     'zab_leg_02_31', 'IBS_POST7', 'age2', 'ROE91', 'endocr_02_28', 'STENOK_AN5',
                                     'NA_KB96', 'n_p_ecg_p_03_66', 'D_AD_ORIT38', 'L_BLOOD90', 'MP_TP_POST41',
                                     'ritm_ecg_p_01_50', 'S_AD_ORIT37', 'S_AD_KBRIG35', 'Na_BLOOD86', 'K_BLOOD84',
                                     'endocr_01_27', 'ritm_ecg_p_07_54', 'ALT_BLOOD87', 'AST_BLOOD88', 'nr04_17',
                                     'IM_PG_P49', 'ritm_ecg_p_02_51', 'LID_S_n106', 'post_im48', 'n_p_ecg_p_07_70',
                                     'ritm_ecg_p_08_55', 'ritm_ecg_p_04_52', 'n_r_ecg_p_01_56', 'n_p_ecg_p_06_69'],
                            "REC_IM122NA60":['gender3', 'L_BLOOD90', 'age2', 'NA_KB96', 'S_AD_ORIT37', 'TIME_B_S92',
                                         'endocr_01_27', 'LID_KB98', 'NITR_S99', 'zab_leg_01_30', 'n_p_ecg_p_11_74',
                                         'IBS_POST7', 'LID_S_n106', 'ZSN_A12', 'n_p_ecg_p_07_70', 'B_BLOK_S_n107',
                                         'zab_leg_02_31', 'inf_im47', 'nr03_16', 'n_p_ecg_p_06_69', 'n_p_ecg_p_03_66',
                                         'n_p_ecg_p_12_75', 'TIKL_S_n111', 'SIM_GIPERT10'],
                            "REC_IM122":['STENOK_AN5', 'L_BLOOD90', 'TIME_B_S92', 'gender3', 'Na_BLOOD86', 'S_AD_KBRIG35',
                                     'GEPAR_S_n109', 'n_p_ecg_p_07_70', 'AST_BLOOD88', 'NA_KB96', 'ritm_ecg_p_01_50',
                                     'LID_KB98', 'ASP_S_n110', 'LID_S_n106', 'n_r_ecg_p_03_58', 'ant_im45',
                                     'endocr_01_27', 'n_r_ecg_p_06_61', 'n_r_ecg_p_04_59', 'IBS_POST7', 'IBS_NASL8',
                                     'ritm_ecg_p_02_51', 'B_BLOK_S_n107', 'ritm_ecg_p_04_52']},
                        "thyroid":{
                            "targetNA60":['17', '20', '19', '1', '0', '9', '2', '6'],
                            "21":['17', '20', '19', '18', '2', '1', '0', '7', '13', '6', '12', '3']},
                        "thyroidNA20":{
                            "targetNA60":['17', '19', '20', '18', '9', '1', '2', '0', '6', '5', '8', '4'],
                            "21":['20', '17', '18', '19', '2', '0', '1', '8', '9']},
                        "thyroidNA10":{
                            "targetNA60":['20', '17', '19', '0', '18', '1', '3', '9', '12', '13', '4'],
                            "21":['20', '17', '18', '2', '19', '1', '0', '11', '7', '6', '4']},
                        "thyroidNA30":{
                            "targetNA60":['17', '18', '19', '20', '2', '7', '5', '1'],
                            "21":['18', '17', '20', '19', '0', '2', '1', '9', '10', '8']},
                        "diabetes":{
                            "readmitted_yesno":['discharge_disposition_id', 'number_inpatient', 'admission_source_id',
                                                 'diag_1', 'diag_2', 'medical_specialty', 'payer_code',
                                                 'number_emergency', 'admission_type_id', 'diabetesMed', 'age',
                                                 'number_outpatient', 'diag_3', 'num_lab_procedures',
                                                 'time_in_hospital', 'weight', 'number_diagnoses', 'num_medications',
                                                 'num_procedures', 'insulin'],
                            "readmitted_yesnoNA60":['discharge_disposition_id', 'number_inpatient',
                                                    'admission_source_id', 'number_emergency', 'diag_1', 'payer_code',
                                                    'weight', 'medical_specialty', 'number_diagnoses',
                                                    'admission_type_id', 'number_outpatient', 'diag_2',
                                                    'num_medications', 'num_procedures', 'time_in_hospital',
                                                    'diabetesMed', 'age', 'insulin', 'A1Cresult',
                                                    'nateglinide', 'tolazamide']
                        },
                        "diabetes_tiny":{
                            "readmitted_yesno":['discharge_disposition_id', 'number_inpatient', 'diag_3', 'diabetesMed', 'num_lab_procedures', 'number_diagnoses', 'num_procedures', 'payer_code', 'number_emergency', 'insulin', 'admission_source_id', 'race', 'medical_specialty', 'admission_type_id', 'gender', 'change', 'A1Cresult', 'weight', 'time_in_hospital', 'number_outpatient', 'max_glu_serum', 'nateglinide'],
                            "readmitted_yesnoNA60":['diag_3', 'admission_type_id', 'number_diagnoses', 'discharge_disposition_id', 'number_inpatient', 'A1Cresult', 'change', 'diabetesMed', 'gender', 'rosiglitazone', 'acarbose', 'glyburide-metformin'],
                        },
}


def feature_selection(train, target, cols):
    """
    :param train: training dataset which will split into train_set and valid_set
    :param target: name of the target variable
    :param cols: column names which should be used as features + target
    :return: nothing
    This function is an iterative process to find the informative feature set.
    The informative feature are identified using permutation importance.
    If the feature importance is positive and the p-value<0.1 then the feature stay in the informative feature set.
    These other features are filtered out.
    Every iteration a new model will be build and the permutation importance will be computed.
    Then the informative feature set will be build and used for the next iteration.
    The iterative process stops when the informative feature set does not change anymore
    """
    train_set = train.loc[train["Set"] == "train",]
    valid_set = train.loc[train["Set"] == "valid",]
    cols = [col for col in cols if not col==target]
    old_cols = cols.copy()
    # dif_cols indicate that there is a difference or not between the old informative feature set and the new one
    dif_cols=True
    while(dif_cols):
        train_set = train_set[[*old_cols,target]]
        valid_set = valid_set[[*old_cols,target]]
        hyperparameters = {
            'GBM': [
                {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
                {},
                'GBMLarge',
            ],
            'CAT': {},
            'XGB': {},
            'RF': [
                {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
                {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
                {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
            ],
            'XT': [
                {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
                {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
                {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
            ],
            'KNN': [
                {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
                {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
            ],
        }
        predictor = TabularPredictor(path=os.path.join(os.getcwd(),"AutogluonLogs"), problem_type="binary",
                                     eval_metric='roc_auc', label=target)
        # train autogluon model
        predictor.fit(train_data=train_set, tuning_data=valid_set,
                      hyperparameters=hyperparameters, auto_stack=False,
                      num_stack_levels=0)
        #compute feature importance
        res = predictor.feature_importance(data=valid_set,subsample_size=1000,num_shuffle_sets=10)

        #filter out uninformative features
        col_names_important=[]
        print("name", end=" ")
        for i in range(res.shape[1]):
            print(res.columns[i], end=" ")
        print("")
        for i in range(res.shape[0]):
            print(res.iloc[i,].name, end=" ")
            if res.iloc[i,0]>0 and res.iloc[i,2]<0.1:
                col_names_important.append(res.iloc[i,].name)
            for k in range(res.shape[1]):
                print(round(res.iloc[i, k],5), end="")
            print("")
        print(col_names_important)
        if set(old_cols)==set(col_names_important):
            dif_cols=False
        else:
            old_cols=col_names_important.copy()
    #print final informative feature set
    print("finish")
    print(target)


def experiment_main(target, dataset_name):
    """
    :param target: name of target variable
    :param dataset_name:  name of dataset
    :return: nothing
    Function load dataset and start the feature_selection function.
    This method run the feature selection task for one target.
    """
    if dataset_name == "myocardial":
        train = pd.read_csv(os.path.join("datasets", "myocardial.csv"), low_memory=False)
        out_cols = ["FIBR_PREDS113", "PREDS_TAH114", "JELUD_TAH115", "FIBR_JELUD116", "A_V_BLOK117", "OTEK_LANC118",
                    "RAZRIV119", "DRESSLER120", "ZSN121", "REC_IM122", "P_IM_STEN123", "LET_IS124", "mortality",
                    "mortalityNA60", "REC_IM122NA60"]
    elif dataset_name == "thyroid":
        train = pd.read_csv(os.path.join("datasets", "thyroid.csv"), low_memory=False)
        out_cols = ["16", "21", "targetNA60"]
    elif dataset_name == "thyroidNA20":
        train = pd.read_csv(os.path.join("datasets", "thyroidNA20.csv"), low_memory=False)
        out_cols = ["16", "21", "targetNA60"]
    elif dataset_name == "thyroidNA10":
        train = pd.read_csv(os.path.join("datasets", "thyroidNA10.csv"), low_memory=False)
        out_cols = ["16", "21", "targetNA60"]
    elif dataset_name == "thyroidNA30":
        train = pd.read_csv(os.path.join("datasets", "thyroidNA30.csv"), low_memory=False)
        out_cols = ["16", "21", "targetNA60"]
    elif dataset_name == "diabetes":
        train = pd.read_csv(os.path.join("datasets", "diabetes.csv"), low_memory=False)
        out_cols = ["readmitted_yesno","readmitted_yesnoNA60"]
    elif dataset_name == "diabetes_tiny":
        train = pd.read_csv(os.path.join("datasets", "diabetes_tiny.csv"), low_memory=False)
        out_cols = ["readmitted_yesno", "readmitted_yesnoNA60"]
    #########################################################################
    out_cols = [col for col in out_cols if not col == target]
    cols = [col for col in train.columns if not col in out_cols]
    if not target in cols:
        cols.append(target)
    train = train[cols]
    cols = [col for col in cols if not col == "Set"]
    train = train.loc[train[target].notna(),]
    feature_selection(train,target,cols)


if __name__ == "__main__":
    # select dataset names
    # select datasets and targets
    # run the feature selection process for all targets
    dataset_names = ["diabetes", "diabetes_tiny", "thyroid", "thyroidNA10","thyroidNA20","thyroidNA30","myocardial"]
    experiments = {"myocardial": ["mortality", "mortalityNA60", "REC_IM122", "REC_IM122NA60"],
                   "thyroid": ["21", "targetNA60"],
                   "thyroidNA10": ["21", "targetNA60"],
                   "thyroidNA20": ["21", "targetNA60"],
                   "thyroidNA30": ["21", "targetNA60"],
                   "diabetes": ["readmitted_yesno", "readmitted_yesnoNA60"],
                   "diabetes_tiny": ["readmitted_yesno", "readmitted_yesnoNA60"]}
    for dataset in dataset_names:
        for target in experiments[dataset]:
            print(f"Dataset name: {dataset}; target: {target}")
            experiment_main(target=target,dataset_name=dataset)
