import os

os.system("pip install beartype==0.17.1 einops==0.7.0 volumentations-3D plotly loguru torchmetrics")
os.system("mkdir preds")
os.system("mkdir plots")
from archs import CTViT_Encoder, extract_encoder
import argparse
from dataset import CTLoader
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import MultiStepLR
import mlflow
import torch.nn.functional as F
from loguru import logger

from torchmetrics.regression import PearsonCorrCoef
from utils import (
    seed_all,
    init_azure,
    query_func,
    filter_empty_input,
    log_metrics,
    get_metrics,
)

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="../saved_weights", type=str)
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--wd", default=0, type=float)
parser.add_argument("--num_epochs", default=20, type=int)
parser.add_argument("--lr_decay", default=10, type=int)
parser.add_argument("--weight_dir", default="../pretrain_weights/", type=str)
parser.add_argument("--train_val_dir", default="../train_val", type=str)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()
print(args.output_dir)

seed_all(args)
blob_service_client = init_azure()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler()
pearson = PearsonCorrCoef()
prior = torch.tensor([0.8424, 1.0 - 0.8424])

train_str, val_str, test_str, ctest_stra_str, wtest_stra_str = (
    "train",
    "val",
    "test",
    "ay_am_ao_sm_sf_so_raa_rw_rb_lvdnm_lvdnf_lvdam_lvdaf_lvsnm_lvsnf_lvsam_lvsaf",
    "ay_am_ao_sm_sf_so_raa_rw_rb_lvdnm_lvdnf_lvdam_lvdaf_lvsnm_lvsnf_lvsam_lvsaf",
)

print(args.train_val_dir)
loader = CTLoader(args.train_val_dir, blob_service_client, args.batch_size, args.num_workers)

#! Cornell prospective study
cornell_prospective_loader, cornell_prospective_df = loader.run_eval("wprospect")
#! Columbia prospective study
columbia_prospective_loader, columbia_prospective_df = loader.run_eval("cprospect")
#! Externel Cornell test
cornell_test_loader, cornell_test_df = loader.run("wtest")
#! Columbia training set
columbia_train_loader, columbia_train_df = loader.run("train")
#! Internel Columbia test
columbia_test_loader, columbia_test_df = loader.run("ctest")
#! Internel Columbia val
columbia_val_loader, columbia_val_df = loader.run('val')
#! Opportunistic screening
opportun_test_loader, opportun_test_df = loader.run_eval("opportun")
# cornell_problem_loader, cornell_problem_df = loader.run_eval('wproblem')
# columbia_problem_loader, columbia_problem_df = loader.run_eval('cproblem')


logger.info("Loader finish")

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
new_pt = extract_encoder(os.path.join(args.weight_dir, "ctvit_pretrained.pt"))
#! 123
# new_pt = torch.load(os.path.join(args.weight_dir, "final_40k", "15_model.pth.tar"))["state_dict"]
#! 42
# new_pt = torch.load(os.path.join(args.weight_dir, "final_40k", "model_best_0.7940022649558552.pth.tar"))["model"]
#! 2024
# new_pt = torch.load(os.path.join(args.weight_dir, "final_40k", "model_best_0.8076645666347813.pth.tar"))["model"]
# #! 2025
# new_pt = torch.load(os.path.join(args.weight_dir, "final_40k", "model_best_0.812722784796209.pth.tar"))["model"]
# #!
# new_pt = torch.load(os.path.join(args.weight_dir, "final_40k", "model_best_0.797650932919924.pth.tar"))["model"]

model.load_state_dict(new_pt)

logger.info("Model loaded")


best_val_auc = 0
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = MultiStepLR(optimizer, milestones=[args.lr_decay], gamma=0.1)


@torch.no_grad()
def val_epoch(dataloader, device, model, epoch, optimizer):
    model.eval()
    state = "Val"
    total_loss = 0.0
    total_preds = torch.tensor([])
    total_labels = torch.tensor([])
    total_lvef = torch.tensor([])
    for batch_idx, (data, labels, weights, lvef, idx) in enumerate(dataloader):
        data, labels, weights, lvef = (
            data.to(device),
            labels.to(device),
            weights.to(device),
            lvef.to(device),
        )
        # data_resampled, label_resampled, lvef_resampled = filter_empty_input(
        #     data, labels, lvef
        # )
        with torch.cuda.amp.autocast():
            output = model.forward(data)
            loss = (F.binary_cross_entropy_with_logits(output, labels.unsqueeze(1).float(), reduction="none")).mean()

        total_loss += loss.item()
        total_preds = torch.cat([total_preds, output.sigmoid().detach().cpu()], dim=0)
        total_labels = torch.cat([total_labels, labels.detach().cpu()], dim=0)
        total_lvef = torch.cat([total_lvef, lvef.detach().cpu()], dim=0)

        if (batch_idx % 1000) == 0:
            print(f"{state}| epoch: {epoch}, step: {batch_idx}, loss: {total_loss / (batch_idx + 1)}")

    return total_preds, total_loss, batch_idx


def train_epoch(dataloader, device, model, epoch, optimizer):
    state = "Train"
    model.train()
    total_loss = 0.0
    total_preds = torch.tensor([])
    total_labels = torch.tensor([])
    total_lvef = torch.tensor([])
    for batch_idx, (data, labels, weights, lvef, idx) in enumerate(dataloader):
        optimizer.zero_grad()
        data, labels, weights, lvef = (
            data.to(device),
            labels.to(device),
            weights.to(device),
            lvef.to(device),
        )

        data_resampled, label_resampled, weights_resampled, lvef_resampled = filter_empty_input(data, labels, weights, lvef)

        # non_empty_data = (torch.amax(data, dim=(1,2,3,4)) <= 0).nonzero()
        with torch.cuda.amp.autocast():
            output = model.forward(data_resampled)
            loss = F.binary_cross_entropy_with_logits(output, label_resampled.unsqueeze(1).float(), reduction="none")
            loss = (loss).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()

        total_loss += loss.item()
        total_preds = torch.cat([total_preds, output.sigmoid().detach().cpu()], dim=0)
        total_labels = torch.cat([total_labels, label_resampled.detach().cpu()], dim=0)
        total_lvef = torch.cat([total_lvef, lvef_resampled.detach().cpu()], dim=0)

        if (batch_idx % 1000) == 0:
            print(f"{state}| epoch: {epoch}, step: {batch_idx}, loss: {total_loss / (batch_idx + 1)}")

    return total_preds, total_loss, batch_idx


mlflow.start_run()
for epoch in range(args.num_epochs):
    train_preds, train_loss, train_batch_idx = train_epoch(columbia_train_loader, device, model, epoch, optimizer)
    # columbia_train_df[f"{epoch}_prob_pred"] = train_preds.numpy()
    # columbia_train_df.to_csv(os.path.join(args.train_val_dir, 'columbia_train_pred.csv'))
    

    # cornell_test_preds, cornell_test_loss, batch_idx = val_epoch(cornell_test_loader, device, model, epoch, optimizer)
    # cornell_test_df[f"{epoch}_prob_pred"] = cornell_test_preds.numpy()
    # cornell_test_df.to_csv(os.path.join(args.train_val_dir, f"cornell_test_pred_{args.seed}.csv"))

    # cornell_pros_preds, cornell_pros_loss, batch_idx = val_epoch(cornell_prospective_loader, device, model, epoch, optimizer)
    # cornell_prospective_df[f"{epoch}_prob_pred"] = cornell_pros_preds.numpy()
    # cornell_prospective_df.to_csv(os.path.join(args.train_val_dir, f"cornell_prospective_pred_latest_{args.seed}.csv"))

    # opportun_preds, opportun_loss, batch_idx = val_epoch(opportun_test_loader, device, model, epoch, optimizer)
    # opportun_test_df[f"{epoch}_prob_pred"] = opportun_preds.numpy()
    # opportun_test_df.to_csv(os.path.join(args.train_val_dir, "opportun_pred_latest.csv"))

    # columbia_pros_preds, columbia_pros_loss, batch_idx = val_epoch(columbia_prospective_loader, device, model, epoch, optimizer)
    # columbia_prospective_df[f"{epoch}_prob_pred"] = columbia_pros_preds.numpy()
    # columbia_prospective_df.to_csv(os.path.join(args.train_val_dir, f"columbia_prospective_pred_latest_{args.seed}.csv"))

    # columbia_test_preds, columbia_test_loss, batch_idx = val_epoch(columbia_test_loader, device, model, epoch, optimizer)
    # columbia_test_df[f"{epoch}_prob_pred"] = columbia_test_preds.numpy()
    # columbia_test_df.to_csv(os.path.join(args.train_val_dir, f"columbia_test_pred_{args.seed}.csv"))

    # columbia_val_preds, columbia_val_loss, batch_idx = val_epoch(columbia_val_loader, device, model, epoch, optimizer)
    # columbia_val_df[f"{epoch}_prob_pred"] = columbia_val_preds.numpy()
    # columbia_val_df.to_csv(os.path.join(args.train_val_dir, f"columbia_val_pred_{args.seed}.csv"))

    state_dict = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}
    torch.save(state_dict, os.path.join(args.weight_dir, "gay9002", 'pyp_exclude', f"{args.seed}_{epoch}_model.pth.tar"))

mlflow.end_run()

print("Done")
