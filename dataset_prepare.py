import os

import pandas as pd
import numpy as np
import json


def prepare_thyroid_dataset():
    dat_train = pd.read_csv(r"datasets\downloaded_datasets\ann-train.data", header=None, sep=" ")
    dat_test = pd.read_csv(r"datasets\downloaded_datasets\ann-test.data", header=None, sep=" ")

    # last two features are empty
    dat_train = dat_train[np.arange(22)]
    dat_test = dat_test[np.arange(22)]

    dat_train["Set"] = "train"
    dat_test["Set"] = "test"

    valid = np.random.choice([True, False], p=[0.2, 0.8], size=dat_train.shape[0])
    missing_values = np.random.choice([True, False], p=[0.6, 0.4], size=dat_train.shape[0])

    print(np.unique(valid,return_counts=True))

    dat_train.loc[valid, "Set"] = "valid"
    dat_train["targetNA60"] = dat_train[21]
    dat_train.loc[missing_values, "targetNA60"] = pd.NA
    dat_test["targetNA60"] = dat_test[21]

    dat_train = dat_train.append(dat_test)
    dat_train = dat_train.reset_index(drop=True)

    for col in dat_train.columns:
        if dat_train[col].dtype == "int64" or dat_train[col].dtype == "object":
            print(col)
            print(np.unique(dat_train.loc[dat_train[col].notna(), col], return_counts=True))
        else:
            print("numerical")
            print(col)
            print(np.unique(dat_train.loc[dat_train[col].notna(), col], return_counts=True))

    # because of too strong imbalance
    out_cols = [14]

    for col in dat_train.columns:
        if dat_train[col].dtype == "int64" and not col == 21:
            dat_train.loc[dat_train[col] == 0, col] = "No"
            dat_train.loc[dat_train[col] == 1, col] = "Yes"
    dat_train.loc[dat_train[21] == 1, 21] = "Yes"
    dat_train.loc[dat_train[21] == 2, 21] = "Yes"
    dat_train.loc[dat_train[21] == 3, 21] = "No"
    dat_train.loc[dat_train["targetNA60"] == 1, "targetNA60"] = "Yes"
    dat_train.loc[dat_train["targetNA60"] == 2, "targetNA60"] = "Yes"
    dat_train.loc[dat_train["targetNA60"] == 3, "targetNA60"] = "No"

    cols = dat_train.columns
    cols = [col for col in cols if col not in out_cols]
    dat_train = dat_train[cols]

    for col in dat_train.columns:
        print(col)
        print(np.unique(dat_train.loc[dat_train[col].notna(), col], return_counts=True))

    dat_train_train = dat_train.loc[dat_train["Set"] == "train", :]
    dat_train_valid = dat_train.loc[dat_train["Set"] == "valid", :]
    dat_train_test = dat_train.loc[dat_train["Set"] == "test", :]
    print(np.unique(dat_train_train.loc[dat_train_train[21].notna(), 21], return_counts=True))
    print(np.unique(dat_train_valid.loc[dat_train_valid[21].notna(), 21], return_counts=True))
    print(np.unique(dat_train_test.loc[dat_train_test[21].notna(), 21], return_counts=True))

    dat10 = dat_train.copy()
    for col in dat10.columns:
        if not col == "21" and not col == "Set" and not col == "targetNA60":
            missing_values = np.random.choice([True, False], p=[0.1, 0.9], size=dat10.shape[0])
            dat10.loc[missing_values, col] = pd.NA

    dat20 = dat_train.copy()
    for col in dat20.columns:
        if not col == "21" and not col == "Set" and not col == "targetNA60":
            missing_values = np.random.choice([True, False], p=[0.2, 0.8], size=dat20.shape[0])
            dat20.loc[missing_values, col] = pd.NA

    dat30 = dat_train.copy()
    for col in dat30.columns:
        if not col == "21" and not col == "Set" and not col == "targetNA60":
            missing_values = np.random.choice([True, False], p=[0.3, 0.7], size=dat30.shape[0])
            dat30.loc[missing_values, col] = pd.NA

    dat_train.to_csv(r"datasets\thyroid.csv", index=False)
    dat10.to_csv(r"datasets\thyroidNA10.csv", index=False)
    dat20.to_csv(r"datasets\thyroidNA20.csv", index=False)
    dat30.to_csv(r"datasets\thyroidNA30.csv", index=False)


def create_hopular_thyroid_dataset(dataset_name,important_features_lists):
    dat = pd.read_csv(os.path.join("datasets",f"{dataset_name}.csv"))
    cols = dat.columns
    cols = [col for col in cols if not col in ["16", "21", "Set", "targetNA60"]]
    labels = dat["21"]
    labelsNA60 = dat["targetNA60"]
    folds = dat["Set"].copy()
    folds[folds != "test"] = 0
    folds[folds == "test"] = 1
    folds_validation = dat["Set"].copy()
    folds_validation[folds_validation != "valid"] = 0
    folds_validation[folds_validation == "valid"] = 1
    dat_x = dat[cols]

    missingNA60 = {}
    for i in range(dat_x.shape[0]):
        mis_col = []
        for col in range(dat_x.shape[1]):
            if pd.isna(dat_x.iloc[i, col]):
                mis_col.append(col)
        if pd.isna(labelsNA60.iloc[i]):
            mis_col.append(dat_x.shape[1])
        missingNA60.update({i: mis_col})

    missing = {}
    for i in range(dat_x.shape[0]):
        mis_col = []
        for col in range(dat_x.shape[1]):
            if pd.isna(dat_x.iloc[i, col]):
                mis_col.append(col)
        if pd.isna(labels.iloc[i]):
            mis_col.append(dat_x.shape[1])
        missing.update({i: mis_col})

    print(dataset_name)
    print("target: 21")
    cat_vars = []
    cont_vars = []
    important_cat_vars = []
    important_cont_vars = []
    important_features_list=important_features_lists[0]
    for i, col in enumerate(dat_x.columns):
        if dat_x[col].dtype == "O":
            cat_vars.append(i)
            if col in important_features_list:
                important_cat_vars.append(i)
        else:
            cont_vars.append(i)
            if col in important_features_list:
                important_cont_vars.append(i)

    print(f"cat vars: {cat_vars}")
    print(f"cont vars: {cont_vars}")
    print(f"important cat vars: {important_cat_vars}")
    print(f"important cont vars: {important_cont_vars}")

    print("cat vars")
    for i in cat_vars:
        print(f"np.asarray([{i}]),")

    print("cont vars")
    for i in cont_vars:
        print(f"np.asarray([{i}]),")

    print("important cat vars")
    for i in important_cat_vars:
        print(f"np.asarray([{i}]),")

    print("important cont vars")
    for i in important_cont_vars:
        print(f"np.asarray([{i}]),")

    print(dataset_name)
    print("target: targetNA60")
    cat_vars = []
    cont_vars = []
    important_cat_vars = []
    important_cont_vars = []
    important_features_list=important_features_lists[1]
    for i, col in enumerate(dat_x.columns):
        if dat_x[col].dtype == "O":
            cat_vars.append(i)
            if col in important_features_list:
                important_cat_vars.append(i)
        else:
            cont_vars.append(i)
            if col in important_features_list:
                important_cont_vars.append(i)

    print(f"cat vars: {cat_vars}")
    print(f"cont vars: {cont_vars}")
    print(f"important cat vars: {important_cat_vars}")
    print(f"important cont vars: {important_cont_vars}")

    print("cat vars")
    for i in cat_vars:
        print(f"np.asarray([{i}]),")

    print("cont vars")
    for i in cont_vars:
        print(f"np.asarray([{i}]),")

    print("important cat vars")
    for i in important_cat_vars:
        print(f"np.asarray([{i}]),")

    print("important cont vars")
    for i in important_cont_vars:
        print(f"np.asarray([{i}]),")

    path_name1 = os.path.join("datasets", "resources", dataset_name)
    os.makedirs(path_name1, exist_ok=True)
    dat_x.to_csv(os.path.join(path_name1,f"{dataset_name}_py.dat"), header=False,
                 index=False)
    folds.to_csv(os.path.join(path_name1, "folds_py.dat"), header=False, index=False)
    labels.to_csv(os.path.join(path_name1, "labels_py.dat"), header=False, index=False)
    folds_validation.to_csv(os.path.join(path_name1, "validation_folds_py.dat"),
                            header=False, index=False)
    with open(os.path.join(path_name1, "missing.json"), "w") as fp:
        json.dump(missing, fp)

    path_name1 = os.path.join("datasets", "resources", f"{dataset_name}small")
    os.makedirs(path_name1, exist_ok=True)
    dat_x.to_csv(os.path.join(path_name1, f"{dataset_name}small_py.dat"), header=False,
                 index=False)
    folds.to_csv(os.path.join(path_name1, "folds_py.dat"), header=False, index=False)
    labels.to_csv(os.path.join(path_name1, "labels_py.dat"), header=False, index=False)
    folds_validation.to_csv(os.path.join(path_name1, "validation_folds_py.dat"),
                            header=False, index=False)
    with open(os.path.join(path_name1, "missing.json"), "w") as fp:
        json.dump(missing, fp)

    path_name1 = os.path.join("datasets", "resources", f"{dataset_name}NA60")
    os.makedirs(path_name1, exist_ok=True)
    dat_x.to_csv(os.path.join(path_name1, f"{dataset_name}NA60_py.dat"), header=False,
                 index=False)
    folds.to_csv(os.path.join(path_name1, "folds_py.dat"), header=False, index=False)
    labelsNA60.to_csv(os.path.join(path_name1, "labels_py.dat"), header=False, index=False)
    folds_validation.to_csv(os.path.join(path_name1, "validation_folds_py.dat"),
                            header=False, index=False)
    with open(os.path.join(path_name1, "missing.json"), "w") as fp:
        json.dump(missingNA60, fp)

    path_name1 = os.path.join("datasets", "resources", f"{dataset_name}NA60small")
    os.makedirs(path_name1, exist_ok=True)
    dat_x.to_csv(os.path.join(path_name1, f"{dataset_name}NA60small_py.dat"), header=False,
                 index=False)
    folds.to_csv(os.path.join(path_name1, "folds_py.dat"), header=False, index=False)
    labelsNA60.to_csv(os.path.join(path_name1, "labels_py.dat"), header=False, index=False)
    folds_validation.to_csv(os.path.join(path_name1, "validation_folds_py.dat"),
                            header=False, index=False)
    with open(os.path.join(path_name1, "missing.json"), "w") as fp:
        json.dump(missingNA60, fp)


def create_hopular_thyroid_datasets():
    create_hopular_thyroid_dataset(dataset_name="thyroid", important_features_lists=[['17', '20', '19', '18', '2', '1', '0', '7', '13', '6', '12', '3'],
                                                                                     ['17', '20', '19', '1', '0', '9', '2', '6']])
    create_hopular_thyroid_dataset(dataset_name="thyroidNA10", important_features_lists=[['20', '17', '18', '2', '19', '1', '0', '11', '7', '6', '4'],
                                                                                         ['20', '17', '19', '0', '18', '1', '3', '9', '12', '13', '4']])
    create_hopular_thyroid_dataset(dataset_name="thyroidNA20", important_features_lists=[['20', '17', '18', '19', '2', '0', '1', '8', '9'],
                                                                                         ['17', '19', '20', '18', '9', '1', '2', '0', '6', '5', '8', '4']])
    create_hopular_thyroid_dataset(dataset_name="thyroidNA30", important_features_lists=[['18', '17', '20', '19', '0', '2', '1', '9', '10', '8'],
                                                                                         ['17', '18', '19', '20', '2', '7', '5', '1']])


def prepare_diabetes_dataset():
    dat = pd.read_csv(r"datasets/downloaded_datasets/diabetic_data.csv")
    cont_vars = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
                 "number_emergency", "number_inpatient", "number_diagnoses"]
    dat = dat.drop_duplicates("patient_nbr")
    dat = dat.reset_index()
    out = ["encounter_id", "patient_nbr"]
    cols = [col for col in dat.columns if col not in out]
    dat = dat[cols]

    for i in range(len(dat["diag_1"])):
        dat.loc[i, "diag_1"] = str(dat.loc[i, "diag_1"]).split(".")[0]
        dat.loc[i, "diag_2"] = str(dat.loc[i, "diag_2"]).split(".")[0]
        dat.loc[i, "diag_3"] = str(dat.loc[i, "diag_3"]).split(".")[0]

    for col in dat.columns:
        dat.loc[dat[col] == "?", col] = pd.NA
        dat.loc[dat[col] == "Unknown/Invalid", col] = pd.NA
        dat.loc[dat[col] == "None", col] = pd.NA

    for col in dat.columns:
        print(col)
        result = np.unique(dat.loc[pd.notna(dat[col]), col], return_counts=True)
        print(result)
        if not col in [*cont_vars, "readmitted", "index"]:
            for i in range(len(result[0])):
                if col in ["diag_1", "diag_2", "diag_3"]:
                    if result[1][i] < 100:
                        dat.loc[dat[col] == result[0][i], col] = pd.NA
                else:
                    if result[1][i] < 14:
                        dat.loc[dat[col] == result[0][i], col] = pd.NA
        print(col)
        result = np.unique(dat.loc[pd.notna(dat[col]), col], return_counts=True)
        print(result)

    out_vars2 = []
    for col in dat.columns:
        result = np.unique(dat.loc[pd.notna(dat[col]), col], return_counts=True)
        if len(result[0]) < 2:
            out_vars2.append(col)
    out_vars2.append("index")
    cols = [col for col in dat.columns if col not in out_vars2]
    dat = dat[cols]

    for col in dat.columns:
        if not col in [*cont_vars, "readmitted"]:
            dat.loc[dat[col].notna(), col] = "N" + dat.loc[dat[col].notna(), col].astype(str)

    random_split = np.random.choice(["train", "valid", "test"], size=len(dat["readmitted"]), replace=True,
                                    p=[0.65, 0.15, 0.20])

    dat["Set"] = random_split

    dat["readmitted_yesno"] = dat["readmitted"]
    dat.loc[dat["readmitted"] == '<30', "readmitted_yesno"] = "YES"
    dat.loc[dat["readmitted"] == '>30', "readmitted_yesno"] = "YES"

    dat["readmitted_yesnoNA60"] = dat["readmitted_yesno"]
    random_missing_label = np.random.choice([True, False], size=len(dat["readmitted_yesno"]), replace=True,
                                            p=[0.60, 0.40])
    dat.loc[random_missing_label, "readmitted_yesnoNA60"] = pd.NA

    cols = [col for col in dat.columns if col not in ["readmitted"]]
    dat = dat[cols]

    dat.to_csv(r"datasets/diabetes.csv", index=False)

    number_training_samples = dat.loc[dat["Set"] == "train",].shape[0]
    training_samples_selection = np.random.choice(["train", "notrain"], p=[0.05, 0.95], size=number_training_samples)
    dat.loc[dat["Set"] == "train", "Set"] = training_samples_selection
    dat = dat.loc[dat["Set"] != "notrain",]

    number_validation_samples = dat.loc[dat["Set"] == "valid",].shape[0]
    validation_samples_selection = np.random.choice(["valid", "novalid"], p=[0.1, 0.9], size=number_validation_samples)
    dat.loc[dat["Set"] == "valid", "Set"] = validation_samples_selection
    dat = dat.loc[dat["Set"] != "novalid",]

    number_test_samples = dat.loc[dat["Set"] == "test",].shape[0]
    test_samples_selection = np.random.choice(["test", "notest"], p=[0.1, 0.9], size=number_test_samples)
    dat.loc[dat["Set"] == "test", "Set"] = test_samples_selection
    dat = dat.loc[dat["Set"] != "notest",]

    print(dat.loc[dat["Set"] == "train", "Set"].shape)
    print(dat.loc[dat["Set"] == "valid", "Set"].shape)
    print(dat.loc[dat["Set"] == "test", "Set"].shape)

    dat.to_csv(r"datasets/diabetes_tiny.csv", index=False)


def create_hopular_diabetes_datasets():
    dat = pd.read_csv(r"datasets/diabetes_tiny.csv", low_memory=False)
    cols = dat.columns
    out_cols = ['Set', 'readmitted_yesno', 'readmitted_yesnoNA60']
    target = 'readmitted_yesno'
    target_dat = dat[target]
    print("train")
    target_dat_tmp = target_dat.loc[dat["Set"] == "train"]
    print(np.unique(target_dat_tmp.loc[target_dat_tmp.notna()], return_counts=True))

    print("valid")
    target_dat_tmp = target_dat.loc[dat["Set"] == "valid"]
    print(np.unique(target_dat_tmp.loc[target_dat_tmp.notna()], return_counts=True))

    print("test")
    target_dat_tmp = target_dat.loc[dat["Set"] == "test"]
    print(np.unique(target_dat_tmp.loc[target_dat_tmp.notna()], return_counts=True))

    cols = [col for col in cols if col not in out_cols]
    x_dat = dat[cols]

    missing = {}
    for i in range(x_dat.shape[0]):
        feature_indices = []
        for k in range(x_dat.shape[1]):
            if pd.isna(x_dat.iloc[i, k]):
                feature_indices.append(k)
        if pd.isna(target_dat.iloc[i]):
            feature_indices.append(x_dat.shape[1])
        missing.update({i: feature_indices})

    cat_vars = []
    cont_vars = []
    important_cat_vars = []
    important_cont_vars = []
    important_features_list = ['discharge_disposition_id', 'number_inpatient', 'diag_3', 'diabetesMed',
                               'num_lab_procedures', 'number_diagnoses', 'num_procedures', 'payer_code',
                               'number_emergency', 'insulin', 'admission_source_id', 'race', 'medical_specialty',
                               'admission_type_id', 'gender', 'change', 'A1Cresult', 'weight', 'time_in_hospital',
                               'number_outpatient', 'max_glu_serum', 'nateglinide']
    for i, col in enumerate(x_dat.columns):
        if x_dat[col].dtype == "O":
            cat_vars.append(i)
            if col in important_features_list:
                important_cat_vars.append(i)
        else:
            cont_vars.append(i)
            if col in important_features_list:
                important_cont_vars.append(i)

    print(f"cat vars: {cat_vars}")
    print(f"cont vars: {cont_vars}")
    print(f"important cat vars: {important_cat_vars}")
    print(f"important cont vars: {important_cont_vars}")

    print("cat vars")
    for i in cat_vars:
        print(f"np.asarray([{i}]),")

    print("cont vars")
    for i in cont_vars:
        print(f"np.asarray([{i}]),")

    print("important cat vars")
    for i in important_cat_vars:
        print(f"np.asarray([{i}]),")

    print("important cont vars")
    for i in important_cont_vars:
        print(f"np.asarray([{i}]),")

    dat["test"] = 0
    dat.loc[dat["Set"] == "test", "test"] = 1
    test_fold = dat["test"]
    dat["valid"] = 0
    dat.loc[dat["Set"] == "valid", "valid"] = 1
    valid_fold = dat["valid"]

    os.makedirs(r"datasets\resources\diabetes", exist_ok=True)
    x_dat.to_csv(r"datasets\resources\diabetes\diabetes_py.dat", header=False, index=False)
    target_dat.to_csv(r"datasets\resources\diabetes\labels_py.dat", header=False, index=False)
    test_fold.to_csv(r"datasets\resources\diabetes\folds_py.dat", header=False, index=False)
    valid_fold.to_csv(r"datasets\resources\diabetes\validation_folds_py.dat", header=False,
                      index=False)
    with open(r"datasets\resources\diabetes\missing.json", "w") as fp:
        json.dump(missing, fp)

    os.makedirs(r"datasets\resources\diabetessmall", exist_ok=True)
    x_dat.to_csv(r"datasets\resources\diabetessmall\diabetessmall_py.dat", header=False,
                 index=False)
    target_dat.to_csv(r"datasets\resources\diabetessmall\labels_py.dat", header=False,
                      index=False)
    test_fold.to_csv(r"datasets\resources\diabetessmall\folds_py.dat", header=False,
                     index=False)
    valid_fold.to_csv(r"datasets\resources\diabetessmall\validation_folds_py.dat",
                      header=False, index=False)
    with open(r"datasets\resources\diabetessmall\missing.json", "w") as fp:
        json.dump(missing, fp)

    target = 'readmitted_yesnoNA60'
    target_dat = dat[target]

    print("train")
    target_dat_tmp = target_dat.loc[dat["Set"] == "train"]
    print(np.unique(target_dat_tmp.loc[target_dat_tmp.notna()], return_counts=True))

    print("valid")
    target_dat_tmp = target_dat.loc[dat["Set"] == "valid"]
    print(np.unique(target_dat_tmp.loc[target_dat_tmp.notna()], return_counts=True))

    print("test")
    target_dat_tmp = target_dat.loc[dat["Set"] == "test"]
    print(np.unique(target_dat_tmp.loc[target_dat_tmp.notna()], return_counts=True))

    missing = {}
    for i in range(x_dat.shape[0]):
        feature_indices = []
        for k in range(x_dat.shape[1]):
            if pd.isna(x_dat.iloc[i, k]):
                feature_indices.append(k)
        if pd.isna(target_dat.iloc[i]):
            feature_indices.append(x_dat.shape[1])
        missing.update({i: feature_indices})

    cat_vars = []
    cont_vars = []
    important_cat_vars = []
    important_cont_vars = []
    important_features_list = ['diag_3', 'admission_type_id', 'number_diagnoses', 'discharge_disposition_id',
                               'number_inpatient', 'A1Cresult', 'change', 'diabetesMed', 'gender', 'rosiglitazone',
                               'acarbose', 'glyburide-metformin']
    for i, col in enumerate(x_dat.columns):
        if x_dat[col].dtype == "O":
            cat_vars.append(i)
            if col in important_features_list:
                important_cat_vars.append(i)
        else:
            cont_vars.append(i)
            if col in important_features_list:
                important_cont_vars.append(i)

    print(f"cat vars: {cat_vars}")
    print(f"cont vars: {cont_vars}")
    print(f"important cat vars: {important_cat_vars}")
    print(f"important cont vars: {important_cont_vars}")

    print("cat vars")
    for i in cat_vars:
        print(f"np.asarray([{i}]),")

    print("cont vars")
    for i in cont_vars:
        print(f"np.asarray([{i}]),")

    print("important cat vars")
    for i in important_cat_vars:
        print(f"np.asarray([{i}]),")

    print("important cont vars")
    for i in important_cont_vars:
        print(f"np.asarray([{i}]),")

    os.makedirs(r"datasets\resources\diabetesNA60", exist_ok=True)
    x_dat.to_csv(r"datasets\resources\diabetesNA60\diabetesNA60_py.dat", header=False,
                 index=False)
    target_dat.to_csv(r"datasets\resources\diabetesNA60\labels_py.dat", header=False,
                      index=False)
    test_fold.to_csv(r"datasets\resources\diabetesNA60\folds_py.dat", header=False,
                     index=False)
    valid_fold.to_csv(r"datasets\resources\diabetesNA60\validation_folds_py.dat",
                      header=False, index=False)
    with open(r"datasets\resources\diabetesNA60\missing.json", "w") as fp:
        json.dump(missing, fp)

    os.makedirs(r"datasets\resources\diabetesNA60small", exist_ok=True)
    x_dat.to_csv(r"datasets\resources\diabetesNA60small\diabetesNA60small_py.dat",
                 header=False, index=False)
    target_dat.to_csv(r"datasets\resources\diabetesNA60small\labels_py.dat", header=False,
                      index=False)
    test_fold.to_csv(r"datasets\resources\diabetesNA60small\folds_py.dat", header=False,
                     index=False)
    valid_fold.to_csv(r"datasets\resources\diabetesNA60small\validation_folds_py.dat",
                      header=False, index=False)
    with open(r"datasets\resources\diabetesNA60small\missing.json", "w") as fp:
        json.dump(missing, fp)


def prepare_myocardial_dataset():
    names = ["ID1", "age2", "gender3", "INF_ANAM4", "STENOK_AN5", "FK_STENOK6", "IBS_POST7", "IBS_NASL8", "GB9",
             "SIM_GIPERT10",
             "DLIT_AG11", "ZSN_A12", "nr11_13", "nr01_14", "nr02_15", "nr03_16", "nr04_17", "nr07_18", "nr08_19",
             "np01_20",
             "np04_21", "np05_22", "np07_23", "np08_24", "np09_25", "np10_26", "endocr_01_27", "endocr_02_28",
             "endocr_03_29",
             "zab_leg_01_30", "zab_leg_02_31", "zab_leg_03_32", "zab_leg_04_33", "zab_leg_06_34", "S_AD_KBRIG35",
             "D_AD_KBRIG36", "S_AD_ORIT37", "D_AD_ORIT38", "O_L_POST39", "K_SH_POST40", "MP_TP_POST41", "SVT_POST42",
             "GT_POST43", "FIB_G_POST44", "ant_im45", "lat_im46", "inf_im47", "post_im48", "IM_PG_P49",
             "ritm_ecg_p_01_50",
             "ritm_ecg_p_02_51", "ritm_ecg_p_04_52", "ritm_ecg_p_06_53", "ritm_ecg_p_07_54", "ritm_ecg_p_08_55",
             "n_r_ecg_p_01_56", "n_r_ecg_p_02_57", "n_r_ecg_p_03_58", "n_r_ecg_p_04_59", "n_r_ecg_p_05_60",
             "n_r_ecg_p_06_61",
             "n_r_ecg_p_08_62", "n_r_ecg_p_09_63", "n_r_ecg_p_10_64", "n_p_ecg_p_01_65", "n_p_ecg_p_03_66",
             "n_p_ecg_p_04_67",
             "n_p_ecg_p_05_68", "n_p_ecg_p_06_69", "n_p_ecg_p_07_70", "n_p_ecg_p_08_71", "n_p_ecg_p_09_72",
             "n_p_ecg_p_10_73",
             "n_p_ecg_p_11_74", "n_p_ecg_p_12_75", "fibr_ter_01_76", "fibr_ter_02_77", "fibr_ter_03_78",
             "fibr_ter_05_79",
             "fibr_ter_06_80", "fibr_ter_07_81", "fibr_ter_08_82", "GIPO_K_83", "K_BLOOD84", "GIPER_Na85", "Na_BLOOD86",
             "ALT_BLOOD87", "AST_BLOOD88", "KFK_BLOOD89", "L_BLOOD90", "ROE91", "TIME_B_S92", "R_AB_1_n93",
             "R_AB_2_n94",
             "R_AB_3_n95", "NA_KB96", "NOT_NA_KB97", "LID_KB98", "NITR_S99", "NA_R_1_n100", "NA_R_2_n101",
             "NA_R_3_n102",
             "NOT_NA_1_n103", "NOT_NA_2_n104", "NOT_NA_3_n105", "LID_S_n106", "B_BLOK_S_n107", "ANT_CA_S_n108",
             "GEPAR_S_n109", "ASP_S_n110", "TIKL_S_n111", "TRENT_S_n112", "FIBR_PREDS113", "PREDS_TAH114",
             "JELUD_TAH115",
             "FIBR_JELUD116", "A_V_BLOK117", "OTEK_LANC118", "RAZRIV119", "DRESSLER120", "ZSN121", "REC_IM122",
             "P_IM_STEN123",
             "LET_IS124"]
    targets = ["FIBR_PREDS113", "PREDS_TAH114", "JELUD_TAH115", "FIBR_JELUD116", "A_V_BLOK117", "OTEK_LANC118",
               "RAZRIV119", "DRESSLER120", "ZSN121", "REC_IM122", "P_IM_STEN123", "LET_IS124", "mortality"]
    out_vars = ["ID1", "R_AB_1_n93", "R_AB_2_n94", "R_AB_3_n95", "NA_R_1_n100", "NA_R_2_n101", "NA_R_3_n102",
                "NOT_NA_1_n103", "NOT_NA_2_n104", "NOT_NA_3_n105"]
    cont_vars = ["age2", "S_AD_KBRIG35", "D_AD_KBRIG36", "S_AD_ORIT37", "D_AD_ORIT38", "K_BLOOD84", "Na_BLOOD86",
                 "ALT_BLOOD87", "AST_BLOOD88", "KFK_BLOOD89", "L_BLOOD90", "ROE91"]
    dat = pd.read_csv(r"datasets/downloaded_datasets/MI.data", header=None, names=names)
    dat["mortality"] = dat["LET_IS124"]
    dat.loc[dat["LET_IS124"] == 0, "mortality"] = "No"
    dat.loc[dat["LET_IS124"] != 0, "mortality"] = "Yes"
    for col in dat.columns:
        print(col)
        result = np.unique(dat.loc[pd.notna(dat[col]), col], return_counts=True)
        print(result)
        if not col in [*cont_vars, "LET_IS124", "ID1"]:
            for i in range(len(result[0])):
                if result[1][i] < 14:
                    dat.loc[dat[col] == result[0][i], col] = "?"
        print(col)
        result = np.unique(dat.loc[pd.notna(dat[col]), col], return_counts=True)
        print(result)

    for col in dat.columns:
        if not col in cont_vars and not col == "mortality":
            dat.loc[:, col] = "N" + dat[col].astype(str)

    dat = dat.replace(["N?", "?"], pd.NA)
    out_vars2 = []
    for col in dat.columns:
        result = np.unique(dat.loc[pd.notna(dat[col]), col], return_counts=True)
        if len(result[0]) < 2:
            out_vars2.append(col)

    cols = [name for name in [*names, "mortality"] if not name in [*out_vars, *out_vars2]]
    dat = dat[cols]
    random_split = np.random.choice(["train", "valid", "test"], size=len(dat["mortality"]), replace=True,
                                    p=[0.65, 0.15, 0.20])
    dat["Set"] = random_split
    dat["REC_IM122NA60"] = dat["REC_IM122"]
    dat["mortalityNA60"] = dat["mortality"]
    random_missing = np.random.choice([True, False], size=len(dat["REC_IM122"]), replace=True, p=[0.4, 0.6])
    dat.loc[random_missing, "REC_IM122NA60"] = pd.NA
    dat.loc[random_missing, "mortalityNA60"] = pd.NA
    dat.to_csv(r"datasets/myocardial.csv", index=False)


def create_hopular_myocardial_dataset(target,name1,important_features):
    dat = pd.read_csv(r"datasets/myocardial.csv")
    targets = ["FIBR_PREDS113", "PREDS_TAH114", "JELUD_TAH115", "FIBR_JELUD116", "A_V_BLOK117", "OTEK_LANC118",
               "RAZRIV119", "DRESSLER120", "ZSN121", "REC_IM122", "P_IM_STEN123", "LET_IS124", "mortality",
               'ZSN121NA60', 'REC_IM122NA60', 'mortalityNA60']
    cont_vars = ["age2", "S_AD_KBRIG35", "D_AD_KBRIG36", "S_AD_ORIT37", "D_AD_ORIT38", "K_BLOOD84", "Na_BLOOD86",
                 "ALT_BLOOD87", "AST_BLOOD88", "KFK_BLOOD89", "L_BLOOD90", "ROE91"]
    cols = dat.columns
    cols = [col for col in cols if col not in [*targets, "Set"]]
    cols_cat = [col for col in cols if col not in cont_vars]
    cols = [*cont_vars, *cols_cat]
    cols = [*cols, 'mortality', 'mortalityNA60', 'REC_IM122', 'REC_IM122NA60', "Set"]
    dat = dat[cols]

    cols2 = [col for col in cols if col not in ['mortality', 'mortalityNA60', 'REC_IM122', 'REC_IM122NA60', "Set"]]
    cols2 = [*cols2, target]
    missing = {}
    missing_entries = False
    for i in range(dat.shape[0]):
        mis_col = []
        for k, col in enumerate(cols2):
            if pd.isna(dat[col].iloc[i]):
                mis_col.append(k)
                missing_entries = True
        missing.update({i: mis_col})

    x_cols = [col for col in cols if col not in ['mortality', 'mortalityNA60', 'REC_IM122', 'REC_IM122NA60', "Set"]]
    dat_x = dat[x_cols]
    labels = dat[target]
    folds = dat["Set"].copy()
    folds[folds != "test"] = 0
    folds[folds == "test"] = 1
    folds_validation = dat["Set"].copy()
    folds_validation[folds_validation != "valid"] = 0
    folds_validation[folds_validation == "valid"] = 1

    indices = []
    for i, col in enumerate(x_cols):
        if col in important_features:
            indices.append(i)
            print(f"np.asarray([{i}]),")
    print(indices)

    path1=os.path.join("datasets","resources",name1)
    os.makedirs(path1, exist_ok=True)
    dat_x.to_csv(os.path.join(path1,f"{name1}_py.dat"), header=False,
                 index=False)
    folds.to_csv(os.path.join(path1,"folds_py.dat"), header=False, index=False)
    labels.to_csv(os.path.join(path1,"labels_py.dat"), header=False, index=False)
    folds_validation.to_csv(os.path.join(path1,"validation_folds_py.dat"),
                            header=False, index=False)
    if missing_entries:
        with open(os.path.join(path1,"missing.json"), "w") as fp:
            json.dump(missing, fp)

    path1 = os.path.join("datasets", "resources", f"{name1}small")
    os.makedirs(path1, exist_ok=True)
    dat_x.to_csv(os.path.join(path1, f"{name1}small_py.dat"), header=False,
                 index=False)
    folds.to_csv(os.path.join(path1, "folds_py.dat"), header=False, index=False)
    labels.to_csv(os.path.join(path1, "labels_py.dat"), header=False, index=False)
    folds_validation.to_csv(os.path.join(path1, "validation_folds_py.dat"),
                            header=False, index=False)
    if missing_entries:
        with open(os.path.join(path1, "missing.json"), "w") as fp:
            json.dump(missing, fp)


def create_hopular_myocardial_datasets():
    important_features = ['ZSN_A12', 'n_p_ecg_p_12_75', 'K_SH_POST40', 'inf_im47', 'TIME_B_S92',
                          'zab_leg_02_31', 'IBS_POST7', 'age2', 'ROE91', 'endocr_02_28', 'STENOK_AN5',
                          'NA_KB96', 'n_p_ecg_p_03_66', 'D_AD_ORIT38', 'L_BLOOD90', 'MP_TP_POST41',
                          'ritm_ecg_p_01_50', 'S_AD_ORIT37', 'S_AD_KBRIG35', 'Na_BLOOD86', 'K_BLOOD84',
                          'endocr_01_27', 'ritm_ecg_p_07_54', 'ALT_BLOOD87', 'AST_BLOOD88', 'nr04_17',
                          'IM_PG_P49', 'ritm_ecg_p_02_51', 'LID_S_n106', 'post_im48', 'n_p_ecg_p_07_70',
                          'ritm_ecg_p_08_55', 'ritm_ecg_p_04_52', 'n_r_ecg_p_01_56', 'n_p_ecg_p_06_69']
    create_hopular_myocardial_dataset(target="mortality", name1="myocardial", important_features=important_features)
    important_features = ['ZSN_A12', 'inf_im47', 'ANT_CA_S_n108', 'zab_leg_02_31', 'age2', 'lat_im46',
                          'K_SH_POST40', 'S_AD_KBRIG35', 'IBS_POST7', 'NA_KB96', 'STENOK_AN5',
                          'n_p_ecg_p_12_75', 'AST_BLOOD88', 'post_im48', 'LID_KB98', 'endocr_01_27',
                          'ritm_ecg_p_02_51', 'O_L_POST39', 'nr04_17', 'ritm_ecg_p_04_52',
                          'n_p_ecg_p_06_69', 'S_AD_ORIT37', 'ritm_ecg_p_01_50', 'ritm_ecg_p_07_54',
                          'INF_ANAM4', 'GEPAR_S_n109', 'n_r_ecg_p_05_60', 'SIM_GIPERT10', 'nr03_16',
                          'gender3', 'nr02_15', 'nr11_13']
    create_hopular_myocardial_dataset(target="mortalityNA60", name1="myocardialNA60", important_features=important_features)
    important_features = ['STENOK_AN5', 'L_BLOOD90', 'TIME_B_S92', 'gender3', 'Na_BLOOD86', 'S_AD_KBRIG35',
                          'GEPAR_S_n109', 'n_p_ecg_p_07_70', 'AST_BLOOD88', 'NA_KB96', 'ritm_ecg_p_01_50',
                          'LID_KB98', 'ASP_S_n110', 'LID_S_n106', 'n_r_ecg_p_03_58', 'ant_im45',
                          'endocr_01_27', 'n_r_ecg_p_06_61', 'n_r_ecg_p_04_59', 'IBS_POST7', 'IBS_NASL8',
                          'ritm_ecg_p_02_51', 'B_BLOK_S_n107', 'ritm_ecg_p_04_52']
    create_hopular_myocardial_dataset(target="REC_IM122", name1="myocardialrec",
                                      important_features=important_features)
    important_features = ['gender3', 'L_BLOOD90', 'age2', 'NA_KB96', 'S_AD_ORIT37', 'TIME_B_S92',
                          'endocr_01_27', 'LID_KB98', 'NITR_S99', 'zab_leg_01_30', 'n_p_ecg_p_11_74',
                          'IBS_POST7', 'LID_S_n106', 'ZSN_A12', 'n_p_ecg_p_07_70', 'B_BLOK_S_n107',
                          'zab_leg_02_31', 'inf_im47', 'nr03_16', 'n_p_ecg_p_06_69', 'n_p_ecg_p_03_66',
                          'n_p_ecg_p_12_75', 'TIKL_S_n111', 'SIM_GIPERT10']
    create_hopular_myocardial_dataset(target="REC_IM122NA60", name1="myocardialrecNA60",
                                      important_features=important_features)


if __name__ == "__main__":
    prepare_thyroid_dataset()
    prepare_diabetes_dataset()
    prepare_myocardial_dataset()
    create_hopular_thyroid_datasets()
    create_hopular_myocardial_datasets()
    create_hopular_diabetes_datasets()