experiment_list=[]
targets=["thyroid_small",
         "thyroidNA60_small",
         "thyroidNA20_small",
         "thyroidNA20NA60_small",
         "myocardial_mortality_small",
         "myocardial_mortality_NA60_small",
         "myocardial_REC_IM_small",
         "myocardial_REC_IM_NA60_small",
         "diabetes_small",
         "diabetesNA60_small",
         "thyroid",
         "thyroidNA60",
         "thyroidNA20",
         "thyroidNA20NA60",
         "myocardial_mortality",
         "myocardial_mortality_NA60",
         "myocardial_REC_IM",
         "myocardial_REC_IM_NA60",
         "diabetes",
         "diabetesNA60",
         "thyroidNA30",
         "thyroidNA30NA60",
         "thyroidNA30_small",
         "thyroidNA30NA60_small",
         "thyroidNA10",
         "thyroidNA10NA60",
         "thyroidNA10_small",
         "thyroidNA10NA60_small",
         "diabetes_tiny",
         "diabetesNA60_tiny",
         "diabetes_small_tiny",
         "diabetesNA60_small_tiny"
         ]
dataset_name_list=["dataset_thyroid_small",
                   "dataset_thyroidNA60_small",
                   "dataset_thyroidNA20_small",
                   "dataset_thyroidNA20NA60_small",
                   "dataset_myocardial_mortality_small",
                   "dataset_myocardial_mortality_NA60_small",
                   "dataset_myocardial_REC_IM_small",
                   "dataset_myocardial_REC_IM_NA60_small",
                   "dataset_diabetes_small",
                   "dataset_diabetesNA60_small",
                   "dataset_thyroid",
                   "dataset_thyroidNA60",
                   "dataset_thyroidNA20",
                   "dataset_thyroidNA20NA60",
                   "dataset_myocardial_mortality",
                   "dataset_myocardial_mortality_NA60",
                   "dataset_myocardial_REC_IM",
                   "dataset_myocardial_REC_IM_NA60",
                   "dataset_diabetes",
                   "dataset_diabetesNA60",
                   "dataset_thyroidNA30",
                   "dataset_thyroidNA30NA60",
                   "dataset_thyroidNA30_small",
                   "dataset_thyroidNA30NA60_small",
                   "dataset_thyroidNA10",
                   "dataset_thyroidNA10NA60",
                   "dataset_thyroidNA10_small",
                   "dataset_thyroidNA10NA60_small",
                   "dataset_diabetes_tiny",
                   "dataset_diabetesNA60_tiny",
                   "dataset_diabetes_small_tiny",
                   "dataset_diabetesNA60_small_tiny"
                   ]

for target, dataset_name in zip(targets,dataset_name_list):
    mixup_lams=[0.2,0.8,0.9]
    mlp_dims=[1000,512,256,128,64,32,16,8]
    lam0_list=[0.5,1,2]
    emb=32
    original_experiment = {"model_name":target,
                           "dataset": dataset_name,
                           "embedding_size": emb,
                           "transformer_depth": 1,
                           "ff_dropout": 0.8,
                           "pretrain": True,
                           "pretrain_epochs": 2, #100
                           "lam0": 0.5,
                           "mlp_dim": 1000,
                           "wd": 0.01,
                           "no_training": 0,
                           "mixup_lam": 0.2,
                           "ensemble": 0,
                           "attention_heads": 8,
                           "batchsize": 256,
                           "attentiontype": 'colrow',
                           "epochs": 2, #50
                           "pt_tasks": ['contrastive', 'denoising']}

    for lam_mixup in mixup_lams:
        for lam0 in lam0_list:

            experiment = original_experiment.copy()
            experiment["pretrain"] = True
            experiment["no_training"] = 1
            experiment["mixup_lam"] = lam_mixup
            experiment["lam0"] = lam0
            experiment["transformer_depth"] = 1
            experiment["attentiontype"] = 'colrow'
            experiment["ff_dropout"] = 0.8
            experiment_list.append(experiment)

            for mlp_dim in mlp_dims:
                experiment = original_experiment.copy()
                experiment["pretrain"] = False
                experiment["no_training"] = 0
                experiment["mixup_lam"] = lam_mixup
                experiment["lam0"] = lam0
                experiment["transformer_depth"] = 1
                experiment["attentiontype"] = 'colrow'
                experiment["ff_dropout"] = 0.8
                experiment["mlp_dim"]=mlp_dim
                experiment["ensemble"] = 0
                experiment_list.append(experiment)

                experiment = original_experiment.copy()
                experiment["pretrain"] = False
                experiment["no_training"] = 0
                experiment["mixup_lam"] = lam_mixup
                experiment["lam0"] = lam0
                experiment["transformer_depth"] = 1
                experiment["attentiontype"] = 'colrow'
                experiment["ff_dropout"] = 0.8
                experiment["mlp_dim"] = mlp_dim
                experiment["ensemble"] = 1
                experiment_list.append(experiment)