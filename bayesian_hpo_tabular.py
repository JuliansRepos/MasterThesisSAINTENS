import os.path
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


"""
The bayesian HPO implementation is oriented to this article:
https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e

Several kaggle projects used such an implementation:
https://www.kaggle.com/code/nandakishorejoshi/datefruit-catboost-with-hyperopt?source=post_page-----5d352e30778d--------------------------------
https://www.kaggle.com/code/konstantinsuloevjr/averaging-models-and-hyperopt-tuning
"""


class BayesianHPO(object):
    """
    Bayesian HPO object
    """
    def __init__(self, x_train, x_test, y_train, y_test):
        """
        :param x_train: training dataset only with features
        :param x_test: training dataset only with target
        :param y_train: validation dataset only with features
        :param y_test: validation dataset only with target
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def search(self, func_name, space, trials, algo, max_evals):
        """
        :param func_name: function name (xgb_cls or lgb_cls or cat_cls)
        :param space: hyperparameter space is defined in the hyperparameter lists
        :param trials: Trials object
        :param algo: algorithm for HPO
        :param max_evals: number of HPO iterations
        :return: results, trials
        Function which find good hyperparameter for GBDT model with Bayesian HPO
        """
        fn = getattr(self, func_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,'exception': str(e)}
        return result, trials

    def xgb_cls(self, para):
        cls = xgb.XGBClassifier(**para['cls_params'])
        return self.train_cls(cls, para)

    def lgb_cls(self, para):
        cls = lgb.LGBMClassifier(**para['cls_params'])
        return self.train_cls(cls, para)

    def cat_cls(self, para):
        cls = cat.CatBoostClassifier(**para['cls_params'])
        return self.train_cls(cls, para)

    def train_cls(self, cls, para):
        """
        :param cls: GBDT object
        :param para: Hyperparameter Liste
        :return: dictionary with loss
        Train GBDT model
        """
        cls.fit(self.x_train, self.y_train,
                eval_set=[(self.x_test, self.y_test)], **para['fit_params'])
        pred = cls.predict_proba(self.x_test)[:,1]
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}

    def evaluate_cls(self, gbdt, para, x_test2, y_test2, dataset_name, target, size):
        """
        :param gbdt: name of GBDT
        :param para: hyperparameter dictionary
        :param x_test2: Test dataset only with features
        :param y_test2: Test dataset only with target
        :param dataset_name: dataset name
        :param target: target name
        :param size: is big (all features) or small (only informative features)
        :return: AUROC on test set and AUROC on validation set
        """
        if gbdt == "cat":
            cls = cat.CatBoostClassifier(**para['cls_params'])
        elif gbdt == "lgb":
            cls = lgb.LGBMClassifier(**para['cls_params'])
        elif gbdt == "xgb":
            cls = xgb.XGBClassifier(**para['cls_params'])
        else:
            print("Wrong GBDT")
        cls.fit(self.x_train, self.y_train, eval_set=[(self.x_test, self.y_test)], **para['fit_params'])
        if gbdt != "lgb":
            cls.save_model(f"models_log/{gbdt}_{dataset_name}_{target}_{size}")
        pred = cls.predict_proba(self.x_test)
        auc_valid = roc_auc_score(self.y_test, pred[:, 1])
        pred = cls.predict_proba(x_test2)
        auc = roc_auc_score(y_test2, pred[:, 1])
        print(f"test: {auc}")
        print(f"valid: {auc_valid}")
        return auc, auc_valid


def load_dataset(dataset_name, target, important_features=False, cat_features=None, para_lists=None):
    """
    :param dataset_name: name of dataset
    :param target: name of target
    :param important_features: if True then important feature set will be used
    :param cat_features: name of categorical features
    :param para_lists:  List of hyperparameters for HPO
    :return: x_train, y_train, x_valid, y_valid, x_test, y_test
    This function load dataset and split it into training, validation and test set
    """
    if dataset_name=="thyroid":
        train = pd.read_csv(os.path.join("datasets","thyroid.csv"), low_memory=False)
        out_cols = ["16", "21", "targetNA60", "Set"]
        if important_features:
            if target == "21":
                cols = ['17', '20', '19', '18', '2', '1', '0', '7', '13', '6', '12', '3']
            elif target == "targetNA60":
                cols = ['17', '20', '19', '1', '0', '9', '2', '6']
            cols = [*cols, *out_cols]
            train = train[cols]
    elif dataset_name=="thyroidNA10":
        train = pd.read_csv(os.path.join("datasets","thyroidNA10.csv"), low_memory=False)
        out_cols = ["16", "21", "targetNA60", "Set"]
        if important_features:
            if target == "21":
                cols = ['20', '17', '18', '2', '19', '1', '0', '11', '7', '6', '4']
            elif target == "targetNA60":
                cols = ['20', '17', '19', '0', '18', '1', '3', '9', '12', '13', '4']
            cols = [*cols, *out_cols]
            train = train[cols]
    elif dataset_name=="thyroidNA20":
        train = pd.read_csv(os.path.join("datasets","thyroidNA20.csv"), low_memory=False)
        out_cols = ["16", "21", "targetNA60", "Set"]
        if important_features:
            if target == "21":
                cols = ['20', '17', '18', '19', '2', '0', '1', '8', '9']
            elif target == "targetNA60":
                cols = ['17', '19', '20', '18', '9', '1', '2', '0', '6', '5', '8', '4']
            cols = [*cols, *out_cols]
            train = train[cols]
    elif dataset_name=="thyroidNA30":
        train = pd.read_csv(os.path.join("datasets","thyroidNA30.csv"), low_memory=False)
        out_cols = ["16", "21", "targetNA60", "Set"]
        if important_features:
            if target == "21":
                cols = ['18', '17', '20', '19', '0', '2', '1', '9', '10', '8']
            elif target == "targetNA60":
                cols = ['17', '18', '19', '20', '2', '7', '5', '1']
            cols = [*cols, *out_cols]
            train = train[cols]
    elif dataset_name=="diabetes":
        train = pd.read_csv(os.path.join("datasets", "diabetes.csv"), low_memory=False)
        out_cols = ["readmitted_yesno", "readmitted_yesnoNA60", "Set"]
        if important_features:
            if target=="readmitted_yesno":
                cols = ['discharge_disposition_id', 'number_inpatient', 'admission_source_id',
                                                 'diag_1', 'diag_2', 'medical_specialty', 'payer_code',
                                                 'number_emergency', 'admission_type_id', 'diabetesMed', 'age',
                                                 'number_outpatient', 'diag_3', 'num_lab_procedures',
                                                 'time_in_hospital', 'weight', 'number_diagnoses', 'num_medications',
                                                 'num_procedures', 'insulin']
            elif target=="readmitted_yesnoNA60":
                cols = ['discharge_disposition_id', 'number_inpatient',
                                                    'admission_source_id', 'number_emergency', 'diag_1', 'payer_code',
                                                    'weight', 'medical_specialty', 'number_diagnoses',
                                                    'admission_type_id', 'number_outpatient', 'diag_2',
                                                    'num_medications', 'num_procedures', 'time_in_hospital',
                                                    'diabetesMed', 'age', 'insulin', 'A1Cresult',
                                                    'nateglinide', 'tolazamide']
            cols = [*cols,*out_cols]
            train = train[cols]
    elif dataset_name=="diabetes_tiny":
        train = pd.read_csv(os.path.join("datasets", "diabetes_tiny.csv"), low_memory=False)
        out_cols = ["readmitted_yesno", "readmitted_yesnoNA60", "Set"]
        if important_features:
            if target=="readmitted_yesno":
                cols = ['discharge_disposition_id', 'number_inpatient', 'diag_3', 'diabetesMed', 'num_lab_procedures', 'number_diagnoses', 'num_procedures', 'payer_code', 'number_emergency', 'insulin', 'admission_source_id', 'race', 'medical_specialty', 'admission_type_id', 'gender', 'change', 'A1Cresult', 'weight', 'time_in_hospital', 'number_outpatient', 'max_glu_serum', 'nateglinide']
            elif target=="readmitted_yesnoNA60":
                cols = ['diag_3', 'admission_type_id', 'number_diagnoses', 'discharge_disposition_id', 'number_inpatient', 'A1Cresult', 'change', 'diabetesMed', 'gender', 'rosiglitazone', 'acarbose', 'glyburide-metformin']
            cols = [*cols,*out_cols]
            train = train[cols]
    elif dataset_name=="myocardial":
        train = pd.read_csv(os.path.join("datasets", "myocardial.csv"), low_memory=False)
        out_cols = ["FIBR_PREDS113", "PREDS_TAH114", "JELUD_TAH115", "FIBR_JELUD116", "A_V_BLOK117", "OTEK_LANC118",
                    "RAZRIV119", "DRESSLER120", "ZSN121", "REC_IM122", "P_IM_STEN123", "LET_IS124", "mortality",
                    "mortalityNA60", "REC_IM122NA60", "Set"]
        if important_features:
            if target == "REC_IM122":
                cols = ['STENOK_AN5', 'L_BLOOD90', 'TIME_B_S92', 'gender3', 'Na_BLOOD86', 'S_AD_KBRIG35',
                        'GEPAR_S_n109', 'n_p_ecg_p_07_70', 'AST_BLOOD88', 'NA_KB96', 'ritm_ecg_p_01_50',
                        'LID_KB98', 'ASP_S_n110', 'LID_S_n106', 'n_r_ecg_p_03_58', 'ant_im45',
                        'endocr_01_27', 'n_r_ecg_p_06_61', 'n_r_ecg_p_04_59', 'IBS_POST7', 'IBS_NASL8',
                        'ritm_ecg_p_02_51', 'B_BLOK_S_n107', 'ritm_ecg_p_04_52']
            elif target == "REC_IM122NA60":
                cols = ['gender3', 'L_BLOOD90', 'age2', 'NA_KB96', 'S_AD_ORIT37', 'TIME_B_S92',
                        'endocr_01_27', 'LID_KB98', 'NITR_S99', 'zab_leg_01_30', 'n_p_ecg_p_11_74',
                        'IBS_POST7', 'LID_S_n106', 'ZSN_A12', 'n_p_ecg_p_07_70', 'B_BLOK_S_n107',
                        'zab_leg_02_31', 'inf_im47', 'nr03_16', 'n_p_ecg_p_06_69', 'n_p_ecg_p_03_66',
                        'n_p_ecg_p_12_75', 'TIKL_S_n111', 'SIM_GIPERT10']
            elif target == "mortality":
                cols = ['ZSN_A12', 'n_p_ecg_p_12_75', 'K_SH_POST40', 'inf_im47', 'TIME_B_S92',
                        'zab_leg_02_31', 'IBS_POST7', 'age2', 'ROE91', 'endocr_02_28', 'STENOK_AN5',
                        'NA_KB96', 'n_p_ecg_p_03_66', 'D_AD_ORIT38', 'L_BLOOD90', 'MP_TP_POST41',
                        'ritm_ecg_p_01_50', 'S_AD_ORIT37', 'S_AD_KBRIG35', 'Na_BLOOD86', 'K_BLOOD84',
                        'endocr_01_27', 'ritm_ecg_p_07_54', 'ALT_BLOOD87', 'AST_BLOOD88', 'nr04_17',
                        'IM_PG_P49', 'ritm_ecg_p_02_51', 'LID_S_n106', 'post_im48', 'n_p_ecg_p_07_70',
                        'ritm_ecg_p_08_55', 'ritm_ecg_p_04_52', 'n_r_ecg_p_01_56', 'n_p_ecg_p_06_69']
            elif target == "mortalityNA60":
                cols = ['ZSN_A12', 'inf_im47', 'ANT_CA_S_n108', 'zab_leg_02_31', 'age2', 'lat_im46',
                        'K_SH_POST40', 'S_AD_KBRIG35', 'IBS_POST7', 'NA_KB96', 'STENOK_AN5',
                        'n_p_ecg_p_12_75', 'AST_BLOOD88', 'post_im48', 'LID_KB98', 'endocr_01_27',
                        'ritm_ecg_p_02_51', 'O_L_POST39', 'nr04_17', 'ritm_ecg_p_04_52',
                        'n_p_ecg_p_06_69', 'S_AD_ORIT37', 'ritm_ecg_p_01_50', 'ritm_ecg_p_07_54',
                        'INF_ANAM4', 'GEPAR_S_n109', 'n_r_ecg_p_05_60', 'SIM_GIPERT10', 'nr03_16',
                        'gender3', 'nr02_15', 'nr11_13']
            cols = [*cols, *out_cols]
            train = train[cols]
    else:
        pass
    # Only use labeled data
    train = train.loc[train[target].notna(),]
    # impute and standardize values
    train = impute_and_standardize(dat=train,target=target,
                                   cols_out=out_cols,cat_features=cat_features, para_lists=para_lists)
    train_set = train.loc[train["Set"]=="train",]
    valid_set = train.loc[train["Set"]=="valid",]
    test_set = train.loc[train["Set"] =="test",]
    cols = [col for col in train.columns if col not in out_cols]
    x_train = train_set[cols]
    y_train = train_set[target]
    x_valid = valid_set[cols]
    y_valid = valid_set[target]
    x_test = test_set[cols]
    y_test = test_set[target]
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def impute_and_standardize(dat, target, cols_out, cat_features=None, para_lists=None):
    """
    :param dat: dataset
    :param target: name of target
    :param cols_out:  cols which should be ignored
    :param cat_features:  name of categorical features
    :param para_lists: List of hyperparameter
    :return: dataset
    If cat_features is None the categorical features are identified automatically.
    Categorical features will transformed into numerical values and missing values get an own category/numerical value.
    Numerical features will be imputed by the mean of the training set and standardized using mean and standard
    deviation of the training set.
    """
    if cat_features is None:
        cols = dat.columns
        cols = [col for col in cols if col not in cols_out]
        cols = [*cols,target]
        cat_features2=[]
        for i, col in enumerate(cols):
            if dat[col].dtype=="O":
                dat.loc[dat[col].isna(),col]="NANA"
                if not col==target:
                    cat_features2.append(i)
                enc = LabelEncoder()
                enc.fit(dat[col])
                dat[col]=enc.transform(dat[col])
            else:
                dat_train = dat.loc[dat["Set"]=="train",]
                mean_train = dat_train.loc[dat_train[col].notna(),col].mean()
                std_train = dat_train.loc[dat_train[col].notna(),col].std()
                dat.loc[dat[col].isna(),col]=mean_train
                dat[col] = (dat[col]-mean_train)/std_train
    else:
        cols = dat.columns
        cols = [col for col in cols if col not in cols_out]
        cols = [*cols, target]
        cat_features2 = []
        for i, col in enumerate(cols):
            if col in cat_features:
                dat.loc[dat[col].isna(),col]="NANA"
                if not col == target:
                    cat_features2.append(i)
                enc = LabelEncoder()
                enc.fit(dat[col])
                dat[col]=enc.transform(dat[col])
            else:
                dat_train = dat.loc[dat["Set"]=="train",]
                mean_train = dat_train.loc[dat_train[col].notna(),col].mean()
                std_train = dat_train.loc[dat_train[col].notna(),col].std()
                dat.loc[dat[col].isna(),col]=mean_train
                dat[col] = (dat[col]-mean_train)/std_train
    # The GBDT models get the indices of categorical features
    if para_lists is not None:
        cat_para, lgb_para, xgb_para = para_lists
        cat_para['fit_params']["cat_features"]=cat_features2
        lgb_para['fit_params']["categorical_feature"]=cat_features2
    return dat


def train_gbdt(datasets, gbdts = ["xgb"]):
    """
    :param datasets: dictionary with dataset names and the targets of each dataset
    :param gbdts: name of GBDT
    :return: nothing
    Function use HPO to find good hyperparameters for Each GBDT model and for each target.
    The good hyperparameters will be used for the final model and this model will be evaluated on the test set.
    The good hyperparameters and the performance of the models will be saved in file.
    """
    for gbdt in gbdts:
        for dataset in datasets.keys():
            for target in datasets[dataset]:
                for important_features in [True, False]:
                    if important_features:
                        important_features_text = "small"
                    else:
                        important_features_text = "big"
                    print(f"gbdt: {gbdt}")
                    print(f"dataset: {dataset}")
                    print(f"target: {target}")
                    print(f"model size: {important_features_text}")
                    #########################################################################
                    # XGB parameters
                    xgb_cls_params = {
                        'learning_rate': hp.loguniform('learning_rate', -7, 0),
                        'max_depth': hp.randint('max_depth', 10) + 1,
                        'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
                        'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
                        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1),
                        'subsample': hp.uniform('subsample', 0.2, 1),
                        'gamma': hp.choice('gamma', [0, hp.loguniform('gamma1', -16, 2), ]),
                        'reg_alpha': hp.choice('reg_alpha', [0, hp.loguniform('reg_alpha1', -16, 2), ]),
                        'reg_lambda': hp.choice('reg_lambda', [0, hp.loguniform('reg_lambda1', -16, 2), ]),
                        'n_estimators': 1000,
                        'booster': 'gbtree',
                        'use_label_encoder': False,
                    }

                    xgb_fit_params = {
                        'eval_metric': 'auc',
                        'verbose': False
                    }

                    xgb_para = dict()
                    xgb_para['cls_params'] = xgb_cls_params
                    xgb_para['fit_params'] = xgb_fit_params
                    xgb_para['loss_func'] = lambda y, pred: (1 - roc_auc_score(y, pred))

                    # LightGBM parameters

                    lgb_cls_params = {
                        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.5)),
                        'max_depth': hp.randint('max_depth', 11) + 1,
                        'min_child_weight': hp.loguniform('min_child_weight', np.log(1), np.log(100)),
                        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                        'subsample': hp.uniform('subsample', 0.5, 1),
                        'reg_alpha': hp.loguniform('reg_alpha', np.log(0.0001), np.log(1)),
                        'reg_lambda': hp.loguniform('reg_lambda', np.log(1), np.log(4)),
                        'boosting_type': hp.choice('boosting_type', ["gbdt", "dart", "goss"]),
                        'num_leaves': hp.randint('num_leaves', 120) + 2,
                        'n_estimators': 1000,
                    }

                    lgb_fit_params = {
                        'eval_metric': 'auc',
                        'verbose': False
                    }

                    lgb_para = dict()
                    lgb_para['cls_params'] = lgb_cls_params
                    lgb_para['fit_params'] = lgb_fit_params
                    lgb_para['loss_func'] = lambda y, pred: (1 - roc_auc_score(y, pred))

                    # CatBoost parameters
                    cat_cls_params = {
                        'learning_rate': hp.loguniform('learning_rate', -5, 0),
                        'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                        'leaf_estimation_iterations': hp.randint('leaf_estimation_iterations', 20) + 1,
                        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1), np.log(10)),
                        'random_strength': hp.randint('random_strength', 20) + 1,
                        'one_hot_max_size': hp.randint('one_hot_max_size', 26),
                        'n_estimators': 1000,
                        'eval_metric': 'AUC',
                        'task_type': "GPU",
                        'boosting_type': 'Ordered',
                    }

                    cat_fit_params = {
                        'verbose': False
                    }

                    cat_para = dict()
                    cat_para['cls_params'] = cat_cls_params
                    cat_para['fit_params'] = cat_fit_params
                    cat_para['loss_func'] = lambda y, pred: (1 - roc_auc_score(y, pred))
                    ###################################################################
                    # load datasets
                    x_train, y_train, x_valid, y_valid, x_test, y_test = load_dataset(dataset_name=dataset,
                                                                                      target=target,
                                                                                      important_features=important_features,
                                                                                      para_lists=(cat_para,lgb_para,xgb_para))
                    obj = BayesianHPO(x_train, x_valid, y_train, y_valid)
                    para_list = {"cat": cat_para,
                                 "lgb": lgb_para,
                                 "xgb": xgb_para}
                    eval_rounds = {"cat": 100,
                                   "lgb": 1000,
                                   "xgb": 1000,
                                   }
                    func_name_list = {"cat": "cat_cls",
                                    "lgb": "lgb_cls",
                                    "xgb": "xgb_cls"}
                    special_para = {"cat": [],
                                    "lgb": ['boosting_type'],
                                    "xgb": ['gamma', 'reg_alpha', 'reg_lambda']}
                    special_procedures = {'boosting_type': ["gbdt", "dart", "goss"],
                                          'gamma': [0, 'gamma1'],
                                          'reg_alpha': [0, 'reg_alpha1'],
                                          'reg_lambda': [0, 'reg_lambda1'],
                                          }
                    # run Bayesian HPO
                    opt_hpo = obj.search(func_name=func_name_list[gbdt], space=para_list[gbdt], trials=Trials(), algo=tpe.suggest,
                                          max_evals=eval_rounds[gbdt])
                    # build dictionary with the good hyperparameter
                    results = opt_hpo[0]
                    hpo_parameter = para_list[gbdt].copy()
                    for k in results.keys():
                        if k not in special_para[gbdt]:
                            hpo_parameter['cls_params'][k] = results[k]
                        else:
                            if gbdt == "lgb":
                                hpo_parameter['cls_params'][k] = special_procedures[k][results[k]]
                            elif gbdt == "xgb":
                                if special_procedures[k][results[k]] == 0:
                                    hpo_parameter['cls_params'][k] = special_procedures[k][results[k]]
                                else:
                                    hpo_parameter['cls_params'][k] = results[special_procedures[k][results[k]]]
                    hpo_parameter['cls_params'] = {k: hpo_parameter['cls_params'][k] for k in hpo_parameter['cls_params'].keys()
                                                   if k not in ['gamma1', 'reg_alpha1', 'reg_lambda1']}
                    for k in hpo_parameter['cls_params'].keys():
                        if gbdt == "xgb" and k == 'max_depth':
                            hpo_parameter['cls_params'][k] += 1
                        elif gbdt == "lgb" and k == 'max_depth':
                            hpo_parameter['cls_params'][k] += 1
                        elif gbdt == "lgb" and k == 'num_leaves':
                            hpo_parameter['cls_params'][k] += 2
                        elif gbdt == "cat" and k == 'leaf_estimation_iterations':
                            hpo_parameter['cls_params'][k] += 1
                        elif gbdt == "cat" and k == 'random_strength':
                            hpo_parameter['cls_params'][k] += 1
                    # Run evaluation of the model with good hyperparameter on the test set
                    auc, auc_valid = obj.evaluate_cls(gbdt, hpo_parameter, x_test, y_test, dataset, target, size=important_features_text)
                    # save performance in file
                    pd.DataFrame({"auc_test": [auc], "auc_valid": [auc_valid]}).to_csv(
                        os.path.join("gbdt_logs",f"{dataset}_{target}_{gbdt}_{important_features_text}.csv"))
                    # save hyperparameters in file
                    with open(os.path.join("gbdt_logs",f"hyperparameters_{dataset}_{target}_{gbdt}_{important_features_text}.txt"), "wt") as filepointer:
                        filepointer.write(f"dataset name: {dataset}")
                        filepointer.write("\n")
                        filepointer.write(f"target name: {target}")
                        filepointer.write("\n")
                        filepointer.write(f"gbdt name: {gbdt}")
                        filepointer.write("\n")
                        filepointer.write(f"valid: {auc_valid}")
                        filepointer.write("\n")
                        filepointer.write(f"test: {auc}")
                        filepointer.write("\n")
                        for k in hpo_parameter['cls_params'].keys():
                            filepointer.write(f"{k}: {str(hpo_parameter['cls_params'][k])}")
                            filepointer.write("\n")


if __name__ == "__main__":
    # select datasets and targets
    # select gbdt methods
    datasets = {"myocardial": ["mortality", "mortalityNA60", "REC_IM122", "REC_IM122NA60"],
                "thyroid": ["21", "targetNA60"],
                "thyroidNA10": ["21", "targetNA60"],
                "thyroidNA20": ["21", "targetNA60"],
                "thyroidNA30": ["21", "targetNA60"],
                "diabetes": ["readmitted_yesno", "readmitted_yesnoNA60"],
                "diabetes_tiny": ["readmitted_yesno", "readmitted_yesnoNA60"]}
    gbdts=["lgb","xgb","cat"]

    train_gbdt(datasets=datasets,gbdts=gbdts)