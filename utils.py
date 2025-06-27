import random
from re import A
import mlflow
import numpy as np
import torch
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from archs import CTViT_Encoder, extract_encoder
import os
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from torchmetrics.regression import PearsonCorrCoef

pearson = PearsonCorrCoef()


def seed_all(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_azure():
    account_url = "https://eus2prdcornellaiechosa.blob.core.windows.net"
    default_credential = DefaultAzureCredential(managed_identity_client_id="b1d972cf-3885-46d8-8021-d553d2871d76")
    blob_service_client = BlobServiceClient(account_url, credential=default_credential)
    return blob_service_client


def create_model(weight_dir, device):
    model = CTViT_Encoder(
        dim=512,
        codebook_size=8192,
        image_size=144,
        patch_size=16,
        temporal_patch_size=2,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    )
    model = model.to(device)
    new_pt = extract_encoder(os.path.join(weight_dir, "ctvit_pretrained.pt"))
    model.load_state_dict(new_pt)
    return model


def filter_empty_input(data, labels, weights, lvef):
    non_empty_input = (torch.amax(data, dim=(1, 2, 3, 4)) > 0).nonzero().squeeze()
    data_resampled = data[non_empty_input]
    label_resampled = labels[non_empty_input]
    weights_resampled = weights[non_empty_input]
    lvef_resampled = lvef[non_empty_input]
    if data_resampled.dim() == 4:
        data_resampled, label_resampled, lvef_resampled = [i.unsqueeze(0) for i in [data_resampled, label_resampled, lvef_resampled]]
    return data_resampled, label_resampled, weights_resampled, lvef_resampled


def save_pred(epoch, total_train_preds, total_train_labels, total_train_lvef, prefix):
    np.save(f"preds/{prefix}_{epoch}.npy", total_train_preds.numpy())
    mlflow.log_artifact(f"preds/{prefix}_{epoch}.npy")
    if prefix == "train" or ("val" in prefix and epoch == 0) or ("test" in prefix and epoch == 0):
        np.save(f"{prefix}_targets{epoch}.npy", total_train_labels.numpy())
        np.save(f"{prefix}_lvef{epoch}.npy", total_train_lvef.numpy())
        mlflow.log_artifact(f"{prefix}_targets{epoch}.npy")
        mlflow.log_artifact(f"{prefix}_lvef{epoch}.npy")


def plot_roc(all_preds, all_labels):
    fpr, tpr, threshold = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    roc_data = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
    random_guess_data = pd.DataFrame({"False Positive Rate": [0, 1], "True Positive Rate": [0, 1]})

    # Create a Plotly Express figure
    fig = px.line(
        roc_data,
        x="False Positive Rate",
        y="True Positive Rate",
        # title=f"ROC Curve (AUC = {roc_auc:.2f})",
        width=700,
        height=500,
    )
    fig.add_scatter(
        x=random_guess_data["False Positive Rate"], y=random_guess_data["True Positive Rate"], mode="lines", line=dict(color="gray", dash="dash"), name="Random Guess"
    )
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", showlegend=True)
    return fig, roc_auc


def get_metrics(epoch, prefix, total_preds, total_labels, total_losses, total_lvef, train_batch_idx):
    metrics = [i(total_labels.numpy(), (total_preds.numpy() >= 0.5).astype(int)) for i in [f1_score, precision_score, recall_score]]
    save_pred(epoch, total_preds, total_labels, total_lvef, prefix)
    train_perason = pearson(total_preds.flatten(), 1 - total_lvef.flatten()).item()
    train_roc, train_auc = plot_roc(total_preds.numpy(), total_labels.numpy())
    train_loss = total_losses / (train_batch_idx + 1)
    return train_auc, metrics, train_loss, train_perason, train_roc


def log_metrics(site_str, train_str, epoch, train_auc, train_metrics, train_loss, train_pearson, train_roc, time_str="all"):
    mlflow.log_metrics(
        {
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} loss": train_loss,
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} auc": train_auc,
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} f1": train_metrics[0],
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} precision": train_metrics[1],
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} recall": train_metrics[2],
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} pearson": train_pearson,
        }
    )
    mlflow.log_figure(
        train_roc,
        f"plots/{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} {epoch} ROC.html",
    )


site_mapping = {"c": "Columbia", "w": "Cornell"}

task_mapping = {
    "train": "Train",
    "val": "Val",
    "test": "Test",
    # "wsm": "Cornell smooth",
    # "wsh": "Cornell sharp",
    "raa": "Race Asian",
    "rw": "Race White",
    "rb": "Race Black",
    "rai": "Race indian/Alaska",
    "lvdnm": "Heart LVD Normal Male",
    "lvdnf": "Heart LVD Normal Female",
    "lvdam": "Heart LVD Abnormal Male",
    "lvdaf": "Heart LVD Abnormal FeMale",
    "lvsnm": "Heart LVS Normal Male",
    "lvsnf": "Heart LVS Normal Female",
    "lvsam": "Heart LVS Abnormal Male",
    "lvsaf": "Heart LVS Abnormal FeMale",
    "sm": "Gender M",
    "sf": "Gender F",
    "so": "Gender O",
    "ay": "Young Age",
    "am": "Middle Age",
    "ao": "Old Age",
    "1d": "CT-ECHO in 1 day",
    "1w": "CT-ECHO in 1 week",
    "1m": "CT-ECHO in 1 month",
    "all": "",
}

race_mapping = {
    "aapi": ["asian", "nat.hawaiian/oth.pacific island"],
    "white": ["white"],
    "black": ["black or african american"],
    "aian": ["american indian or alaska nation"],
}

query_func = {
    "raa": f"race_1 in {race_mapping['aapi']}",
    "rw": f"race_1 in {race_mapping['white']}",
    "rb": f"race_1 in {race_mapping['black']}",
    "ay": "Patients_Age < 40",
    "am": "40 <= Patients_Age < 65",
    "ao": "65 <= Patients_Age < 150",
    "sm": 'Patients_Sex == "M" ',
    "sf": 'Patients_Sex == "F" ',
    "so": 'Patients_Sex == "O" ',
    "lvdnm": "lvd_classification == 'Normal' & Patients_Sex == 'M' ",
    "lvdnf": "lvd_classification == 'Normal' & Patients_Sex == 'F' ",
    "lvdam": "lvd_classification == 'Abnormal' & Patients_Sex == 'M'  ",
    "lvdaf": "lvd_classification == 'Abnormal' & Patients_Sex == 'F'  ",
    "lvsnm": "lvs_classification == 'Normal' & Patients_Sex == 'M' ",
    "lvsnf": "lvs_classification == 'Normal' & Patients_Sex == 'F' ",
    "lvsam": "lvs_classification == 'Abnormal' & Patients_Sex == 'M' ",
    "lvsaf": "lvs_classification == 'Abnormal' & Patients_Sex == 'F' ",
}


def calc_ba(insti_str, query_str, time_str):
    targets = all_targets.query(query_func[query_str])
    time_arg1, time_arg2 = int(time_str[0]), time_str[1].upper()
    targets = targets[(targets["ECHO_study_dttm_y"] - targets["CT_StudyDttm_y"]) <= np.timedelta64(time_arg1, time_arg2)]
    preds = np.load(f"{insti_str}-{query_str}-{time_str}_15.npy")
    return balanced_accuracy_score(targets["labels"].to_numpy(), preds > 0.5)


def ba(columbia, stratify=False):
    for time_str in ["all", "1d", "1w", "1m"]:
        if time_str != "all":
            time_arg1, time_arg2 = int(time_str[0]), time_str[1].upper()
            stratified_val = columbia[(columbia["ECHO_study_dttm_y"] - columbia["CT_StudyDttm_y"]) <= np.timedelta64(time_arg1, time_arg2)]
            stratified_index = (columbia["Path"].isin(stratified_val["Path"])).to_numpy().nonzero()[0]
            tmp = columbia.iloc[stratified_index]
        else:
            tmp = columbia
        if stratify:
            print(
                [
                    balanced_accuracy_score(tmp.query(query_func[i])["labels"].to_numpy(), tmp.query(query_func[i])["pred"].to_numpy() > 0.5)
                    for i in ["sm", "sf", "so", "ay", "am", "ao", "raa", "rw", "rb"]
                ]
            )
        else:
            print([balanced_accuracy_score(tmp["labels"].to_numpy(), tmp["pred"].to_numpy() > 0.5)])
