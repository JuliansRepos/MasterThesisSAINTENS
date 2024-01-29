import torch
from torch import nn
from models.pretrainmodel import SAINT

from data import data_prep,DataSetCatCon

import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters
from augmentations import embed_data_mask
#########################################
from augmentations import embed_data
from utils import eval_model_prediction
from models.model import simple_MLP, EnsembleMLP
from experiments import experiment_list
import pandas as pd
#########################################

import os
import numpy as np


"""
This implementation of SAINT and SAINTENS is based on: https://github.com/somepago/saint/tree/main/old_version

The original SAINT implementation of https://github.com/somepago/saint/tree/main/old_version is under Apache 2.0 license.
"""


default_dict1={
    "dataset":None,
    "cont_embeddings":"MLP",
    "embedding_size":8,
    "transformer_depth":6,
    "attention_heads":8,
    "attention_dropout":0.1,
    "ff_dropout":0.8,
    "attentiontype":"colrow",
    "lr":0.0001,
    "epochs":50,
    "batchsize":256,
    "savemodelroot":'./bestmodels',
    "run_name":"testrun",
    "set_seed":1,
    "pretrain":True,
    "pretrain_epochs":100,
    "pt_tasks":['contrastive','denoising'],
    "pt_aug":['mixup','cutmix'],
    "pt_aug_lam":0.3,
    "mixup_lam":0.9,
    "train_mask_prob":0,
    "pt_projhead_style":"diff",
    "nce_temp":0.7,
    "lam0":10,
    "lam1":10,
    "lam2":1,
    "lam3":1,
    "final_mlp_style":"sep",
    "ensemble":0,
    "mlp_dim":32,
    "wd":0.01,
    "model_name":"modelname",
    "no_training":0,
    "y_dim":2
}
"""
dataset: name of dataset
cont_embeddings: possible options ['MLP','Noemb','pos_singleMLP']; how continuous features will be embedded
embedding_size: embedding size of features
transformer_depth: how many transformer layer
attention_heads: how many attention heads in each transformer
attention_dropout: percentage of attention dropout
ff_dropout: percentage of fully-forward dropout inside of one transformer layer
attentiontype: possible options ['col','colrow','row','justmlp','attn','attnmlp']
lr: learning rate
epochs: number of epochs during the finetuning step
batchsize: batch size
savemodelroot: where the models will be saved
run_name: run name
set_seed: set seed
pretrain: if true the pretraining will be used otherwise it will be skipped
pretrain_epochs: number of pretraining epochs
pt_tasks: possible options ['contrastive','contrastive_sim','denoising']; pretraining tasks
pt_aug: possible options ['mixup','cutmix','gauss_noise']; pretraining augmentation
pt_aug_lam: pretraining augmentation lambda; is the percentage of values which will be exchanged in CutMix
mixup_lam: percentage of how big is the part of the original vector in Mixup and 1-mixup_lam describes how big is the part of another vector out of the same batch
train_mask_prob: is not used
pt_projhead_style: pretraining projection head style; possible options ['diff','same','nohead']
nce_temp: Temperature coefficient for NCE loss
lam0: weighting of contrastive loss
lam1: weighting of contrastive sim loss
lam2: weighting of categorical loss in denoise task
lam3: weighting of continuous loss in denoise task
final_mlp_style: possible options ['common','sep']; if each feature has its own mlp classifier in the end or one common
ensemble: 0 when no MLP Ensemble will be used and 1 when MLP Ensemble Classifier will be used
mlp_dim: size of MLP Classifier/Classifiers in the end in the finetuning step
wd: weight decay
model_name: name of model
no_training: 0 when finetuning step will be done and 1 when no finetuning step will be done
y_dim: Dimension of the target variable
"""


def start_experiment(experiment_dict1, default_dict1):
    """
    prepare the experiment dictionary, set the device, model name and models save path
    :param experiment_dict1: dictionary which includes specific hyperparameter for the current experiments
    :param default_dict1: dictionary which includes the default hyperparameters
    :return: experiment_dict2 (includes all hyperparameter for the current experiment), device, modelsave_path, model_name
    """
    print(experiment_dict1)
    experiment_dict2=default_dict1.copy()
    for k in experiment_dict1.keys():
        experiment_dict2[k]=experiment_dict1[k]

    torch.manual_seed(experiment_dict2["set_seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}.")
    modelsave_path = os.path.join(os.getcwd(), experiment_dict2["savemodelroot"], experiment_dict2["dataset"], experiment_dict2["run_name"])
    os.makedirs(modelsave_path, exist_ok=True)
    model_name = experiment_dict2["model_name"]
    return experiment_dict2,device,modelsave_path,model_name


def pretraining(experiment_dict2, device, modelsave_path, model_name):
    """
    Pretraining of SAINT and SAINTENS models
    :param experiment_dict2: dictionary which includes all hyperparameter for the current experiment
    :param device: used device
    :param modelsave_path: path where models will be saved
    :param model_name: model name
    :return:
    """
    cat_dims, cat_idxs, con_idxs, X_train_pt, y_train_pt, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep(experiment_dict2["dataset"], experiment_dict2["set_seed"])
    ctd = np.array([train_mean, train_std]).astype(np.float32)
    pt_train_ds = DataSetCatCon(X_train_pt, y_train_pt, cat_idxs, ctd, is_pretraining=True)
    pt_trainloader = DataLoader(pt_train_ds, batch_size=experiment_dict2["batchsize"], shuffle=True, num_workers=0, drop_last=True)
    continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

    model = SAINT(
        categories=tuple(cat_dims),
        num_continuous=len(con_idxs),
        dim=experiment_dict2["embedding_size"],
        dim_out=1,
        depth=experiment_dict2["transformer_depth"],
        heads=experiment_dict2["attention_heads"],
        attn_dropout=experiment_dict2["attention_dropout"],
        ff_dropout=experiment_dict2["ff_dropout"],
        mlp_hidden_mults=(4, 2),
        continuous_mean_std=continuous_mean_std,
        cont_embeddings=experiment_dict2["cont_embeddings"],
        attentiontype=experiment_dict2["attentiontype"],
        final_mlp_style=experiment_dict2["final_mlp_style"],
        y_dim=experiment_dict2["y_dim"]
    )

    model.to(device)

    # create logger
    ##########################
    min_loss = None
    pretrain_logger = {"running_loss": [], "denoise_loss": [], "contrastive_loss": []}
    ##########################

    optimizer = optim.AdamW(model.parameters(), lr=experiment_dict2["lr"])
    pt_aug_dict = {
        'noise_type': experiment_dict2["pt_aug"],
        'lambda': experiment_dict2["pt_aug_lam"]
    }
    criterion = nn.CrossEntropyLoss().to(device)
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.MSELoss().to(device)

    print("Pretraining begins!")
    for epoch in range(experiment_dict2["pretrain_epochs"]):
        model.train()
        running_loss = 0.0

        # For logging
        ########################
        denoise_loss = []
        contrastive_loss = []
        ########################

        for i, data in enumerate(pt_trainloader, 0):
            optimizer.zero_grad()
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[
                3].to(device)

            # embed_data_mask function is used to embed both categorical and continuous data.
            if 'cutmix' in experiment_dict2["pt_aug"]:
                from augmentations import add_noise

                x_categ_corr, x_cont_corr = add_noise(x_categ, x_cont, noise_params=pt_aug_dict)

                # I did not use mask embedding
                ##############################
                _, x_categ_enc_2, x_cont_enc_2 = embed_data(x_categ_corr, x_cont_corr, model)
                ##############################

            else:
                _, x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)

            # I did not use mask embedding
            #####################################
            _, x_categ_enc, x_cont_enc = embed_data(x_categ, x_cont, model)
            #####################################
            if 'mixup' in experiment_dict2["pt_aug"]:
                from augmentations import mixup_data
                x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2, lam=experiment_dict2["mixup_lam"])

            loss = 0
            if 'contrastive' in experiment_dict2["pt_tasks"]:
                aug_features_1 = model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1, 2)
                if experiment_dict2["pt_projhead_style"] == 'diff':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp2(aug_features_2)
                elif experiment_dict2["pt_projhead_style"] == 'same':
                    aug_features_1 = model.pt_mlp(aug_features_1)
                    aug_features_2 = model.pt_mlp(aug_features_2)
                else:
                    print('Not using projection head')
                logits_per_aug1 = aug_features_1 @ aug_features_2.t() / experiment_dict2["nce_temp"]
                logits_per_aug2 = aug_features_2 @ aug_features_1.t() / experiment_dict2["nce_temp"]
                targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                loss_1 = criterion(logits_per_aug1, targets)
                loss_2 = criterion(logits_per_aug2, targets)
                loss = experiment_dict2["lam0"] * (loss_1 + loss_2) / 2

                # For logging
                #####################
                contrastive_loss.append((experiment_dict2["lam0"] * (loss_1 + loss_2) / 2).item())
                #####################

            elif 'contrastive_sim' in experiment_dict2["pt_tasks"]:
                aug_features_1 = model.transformer(x_categ_enc, x_cont_enc)
                aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug_features_1 = model.pt_mlp(aug_features_1)
                aug_features_2 = model.pt_mlp2(aug_features_2)
                c1 = aug_features_1 @ aug_features_2.t()
                loss += experiment_dict2["lam1"] * torch.diagonal(-1 * c1).add_(1).pow_(2).sum()
            if 'denoising' in experiment_dict2["pt_tasks"]:
                cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
                con_outs = torch.cat(con_outs, dim=1)

                # Only values which were not NA values in the original sample should be denoised
                ###########################################
                cat_mask = cat_mask.to(dtype=torch.bool)
                con_mask = con_mask.to(dtype=torch.bool)
                l2 = criterion2(con_outs[con_mask], x_cont[con_mask])
                ###########################################

                l1 = 0
                for j in range(1, len(cat_dims) - 1):

                    # Only values which were not NA values in the original sample should be denoised
                    ####################################
                    if cat_mask[:, j].sum() > 0:
                        l1 += criterion1(cat_outs[j][cat_mask[:, j]], x_categ[:, j][cat_mask[:, j]])
                    ####################################

                loss += experiment_dict2["lam2"] * l1 + experiment_dict2["lam3"] * l2

                # For logging
                #####################
                denoise_loss.append((experiment_dict2["lam2"] * l1 + experiment_dict2["lam3"] * l2).item())
                if str(denoise_loss[-1]) == "nan":
                    print("Error")
                #####################

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # For logging and supervise the training process
        #############################################
        print("_________________________________________")
        print(
            f'Epoch: {epoch}, Running Loss: {running_loss}, Denoise Loss: {np.sum(denoise_loss)}, Contrastive Loss: {np.sum(contrastive_loss)}')
        print(
            f'Epoch: {epoch}, Denoise Loss Mean: {np.mean(denoise_loss)}, Denoise Loss Std: {np.std(denoise_loss)}, Contrastive Loss Mean: {np.mean(contrastive_loss)}, Contrastive Loss Std: {np.std(contrastive_loss)}')
        pretrain_logger["running_loss"].append(running_loss)
        pretrain_logger["denoise_loss"].append(np.sum(denoise_loss))
        pretrain_logger["contrastive_loss"].append(np.sum(contrastive_loss))
        torch.save(model.state_dict(),
                   f"{modelsave_path}/pretrain_model_{model_name}_alpha_{experiment_dict2['mixup_lam']}_lam0_{experiment_dict2['lam0']}_emb_{experiment_dict2['embedding_size']}.pth")
        if min_loss is None:
            min_loss = running_loss
            torch.save(model.state_dict(),
                       f"{modelsave_path}/pretrain_bestmodel_{model_name}_alpha_{experiment_dict2['mixup_lam']}_lam0_{experiment_dict2['lam0']}_emb_{experiment_dict2['embedding_size']}.pth")
        elif running_loss < min_loss:
            min_loss = running_loss
            torch.save(model.state_dict(),
                       f"{modelsave_path}/pretrain_bestmodel_{model_name}_alpha_{experiment_dict2['mixup_lam']}_lam0_{experiment_dict2['lam0']}_emb_{experiment_dict2['embedding_size']}.pth")
        #############################################

    # For logging
    ########################
    pd.DataFrame(pretrain_logger).to_csv(
        f"Logs/pretrain_logger_{model_name}_alpha_{experiment_dict2['mixup_lam']}_lam0_{experiment_dict2['lam0']}_emb_{experiment_dict2['embedding_size']}.csv")
    ########################


def finetuning(experiment_dict2, device, modelsave_path, model_name):
    """
    Finetuning step
    SAINTENS only train the MLP Ensemble Classifier and SAINT train the whole model
    The best pretrain model from above will be used
    :param experiment_dict2: dictionary which includes all hyperparameter for the current experiment
    :param device: used device
    :param modelsave_path: path where models will be saved
    :param model_name: model name
    :return:
    """
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep(
        experiment_dict2["dataset"], experiment_dict2["set_seed"])
    continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

    train_ds = DataSetCatCon(X_train, y_train, cat_idxs,continuous_mean_std, is_pretraining=False)
    trainloader = DataLoader(train_ds, batch_size=experiment_dict2["batchsize"], shuffle=True,num_workers=0,drop_last=True)

    valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs,continuous_mean_std, is_pretraining=False)
    validloader = DataLoader(valid_ds, batch_size=experiment_dict2["batchsize"], shuffle=False,num_workers=0,drop_last=False)

    test_ds = DataSetCatCon(X_test, y_test, cat_idxs,continuous_mean_std, is_pretraining=False)
    testloader = DataLoader(test_ds, batch_size=experiment_dict2["batchsize"], shuffle=False,num_workers=0,drop_last=False)

    model = SAINT(
        categories=tuple(cat_dims),
        num_continuous=len(con_idxs),
        dim=experiment_dict2["embedding_size"],
        dim_out=1,
        depth=experiment_dict2["transformer_depth"],
        heads=experiment_dict2["attention_heads"],
        attn_dropout=experiment_dict2["attention_dropout"],
        ff_dropout=experiment_dict2["ff_dropout"],
        mlp_hidden_mults=(4, 2),
        continuous_mean_std=continuous_mean_std,
        cont_embeddings=experiment_dict2["cont_embeddings"],
        attentiontype=experiment_dict2["attentiontype"],
        final_mlp_style=experiment_dict2["final_mlp_style"],
        y_dim=experiment_dict2["y_dim"]
    )

    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    # For logging
    # For skipping pretraining, because some targets can use the same pretraining model
    # Loading the pretrained model
    ##############################
    train_logger = {"train_auc": [], "valid_auc": [], "test_auc": []}

    model.load_state_dict(torch.load(
        f"{modelsave_path}/pretrain_bestmodel_{model_name}_alpha_{experiment_dict2['mixup_lam']}_lam0_{experiment_dict2['lam0']}_emb_{experiment_dict2['embedding_size']}.pth"))
    model = model.to(device)

    best_valid_auroc = 0
    best_test_auroc = 0

    print('Training begins now.')

    # SAINTENS
    # Each classifier gets his own optimizer
    if experiment_dict2["ensemble"]:
        model.classifier = EnsembleMLP(50, model.dim * (model.num_continuous + model.num_categories - 1), experiment_dict2["mlp_dim"])
        model = model.to(device)
        optimizer = []
        for mlp in model.classifier.mlps:
            optimizer.append(optim.AdamW(mlp.parameters(), lr=experiment_dict2["lr"], weight_decay=experiment_dict2["wd"]))
    # SAINT
    else:
        model.mlpfory = simple_MLP([model.dim, experiment_dict2["mlp_dim"], 2])
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=experiment_dict2["lr"], weight_decay=experiment_dict2["wd"])

    # Only the classifier will be trained when the whole representation will be used
    if experiment_dict2["ensemble"]:
        for par in model.transformer.parameters():
            par.requires_grad_(False)
        for par in model.embeds.parameters():
            par.requires_grad_(False)
        for par in model.simple_MLP.parameters():
            par.requires_grad_(False)
    ###########################################

    for epoch in range(experiment_dict2["epochs"]):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            if experiment_dict2["ensemble"]:
                for optimizer_one in optimizer:
                    optimizer_one.zero_grad()
            else:
                optimizer.zero_grad()

            # Mask embedding will not be used
            #########################
            x_categ, x_cont, y = data[0].to(device), data[1].to(device), data[2].to(device=device, dtype=torch.long)
            _, x_categ_enc, x_cont_enc = embed_data(x_categ, x_cont, model)
            #########################

            # We are converting the data to embeddings in the next step
            reps = model.transformer(x_categ_enc, x_cont_enc)

            # For SAINTENS which uses the whole representation
            if experiment_dict2["ensemble"]:
                y_reps = torch.flatten(reps[:, 1:, :], start_dim=1)
                y_outs = model.classifier(y_reps)
                loss = None
                for mlp_out in y_outs:
                    if loss is None:
                        loss = criterion(mlp_out, y[:, 0])
                    else:
                        loss += criterion(mlp_out, y[:, 0])
            else:
                y_reps = reps[:, 0, :]
                y_outs = model.mlpfory(y_reps)
                loss = criterion(y_outs, y[:, 0])

            loss.backward()

            if experiment_dict2["ensemble"]:
                for optimizer_one in optimizer:
                    optimizer_one.step()
            else:
                optimizer.step()

            running_loss += loss.item()

        torch.save(model.state_dict(),
                   f"{modelsave_path}/train_model_{model_name}_alpha_{experiment_dict2['mixup_lam']}_lam0_{experiment_dict2['lam0']}_emb_{experiment_dict2['embedding_size']}_ens_{experiment_dict2['ensemble']}_mlpdim_{experiment_dict2['mlp_dim']}.pth")

        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                train_accuracy, train_auroc = eval_model_prediction(model, trainloader, device, experiment_dict2["ensemble"])
                accuracy, auroc = eval_model_prediction(model, validloader, device, experiment_dict2["ensemble"])
                test_accuracy, test_auroc = eval_model_prediction(model, testloader, device, experiment_dict2["ensemble"])

                print('[EPOCH %d] TRAIN ACCURACY: %.5f, TRAIN AUROC: %.5f' %
                      (epoch + 1, train_accuracy, train_auroc))

                print('[EPOCH %d] VALID ACCURACY: %.5f, VALID AUROC: %.5f' %
                      (epoch + 1, accuracy, auroc))
                print('[EPOCH %d] TEST ACCURACY: %.5f, TEST AUROC: %.5f' %
                      (epoch + 1, test_accuracy, test_auroc))
                if best_valid_auroc < 0.5 and best_valid_auroc > 0:
                    best_valid_auroc = 1 - best_valid_auroc
                auroc_small = False
                if auroc < 0.5:
                    auroc = 1 - auroc
                    auroc_small = True
                if auroc > best_valid_auroc:
                    best_valid_auroc = auroc
                    best_test_auroc = test_auroc

                    torch.save(model.state_dict(),
                               f"{modelsave_path}/train_bestmodel_{model_name}_alpha_{experiment_dict2['mixup_lam']}_lam0_{experiment_dict2['lam0']}_emb_{experiment_dict2['embedding_size']}_ens_{experiment_dict2['ensemble']}_mlpdim_{experiment_dict2['mlp_dim']}.pth")

            # For logging
            ############
            train_logger["train_auc"].append(train_auroc)
            train_logger["valid_auc"].append(auroc)
            train_logger["test_auc"].append(test_auroc)
            print(f"AUC difference: {best_valid_auroc - auroc}")
            if best_valid_auroc < 0.5 and best_valid_auroc > 0:
                best_valid_auroc = 1 - best_valid_auroc
            if auroc < 0.5:
                auroc = 1 - auroc
                auroc_small = True
            if (best_valid_auroc - auroc) > 0.01 and not auroc_small:
                break
            ############

            model.train()

    # For logging
    ################
    train_logger["train_auc"].append(best_test_auroc)
    train_logger["valid_auc"].append(best_valid_auroc)
    train_logger["test_auc"].append(best_test_auroc)
    pd.DataFrame(train_logger).to_csv(
        f"Logs/train_logger_{model_name}_alpha_{experiment_dict2['mixup_lam']}_lam0_{experiment_dict2['lam0']}_emb_{experiment_dict2['embedding_size']}_ens_{experiment_dict2['ensemble']}_mlpdim_{experiment_dict2['mlp_dim']}.csv")
    ################

    total_parameters = count_parameters(model)
    print('TOTAL NUMBER OF PARAMS: %d' % (total_parameters))
    print('AUROC on best model:  %.5f' % (best_test_auroc))


if __name__ == "__main__":
    # experiment_list is defined in experiments.py
    # All the experiments of the experiment_list will be run
    for experiment in experiment_list:
        experiment_dict2,device,modelsave_path,model_name = start_experiment(experiment_dict1=experiment,default_dict1=default_dict1)
        if experiment_dict2["pretrain"]:
            pretraining(experiment_dict2=experiment_dict2, device=device, modelsave_path=modelsave_path, model_name=model_name)
        if not experiment_dict2["no_training"]:
            finetuning(experiment_dict2=experiment_dict2, device=device, modelsave_path=modelsave_path, model_name=model_name)