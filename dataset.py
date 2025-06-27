import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from azure.storage.blob import BlobServiceClient
import numpy as np
import os

from volumentations import *


class CTDataset(Dataset):
    def __init__(self, data_file, transform, blob_service_client: BlobServiceClient):
        self.dataset = data_file
        # self.dataset["ECHO_study_dttm_y"] = pd.to_datetime(self.dataset["ECHO_study_dttm_y"])
        # self.dataset["CT_StudyDttm_y"] = pd.to_datetime(self.dataset["CT_StudyDttm_y"])
        self.blob_service_client = blob_service_client

        # pos_weight = len(data_file) / (data_file["labels"] == 1).sum()
        # neg_weight = len(data_file) / (data_file["labels"] == 0).sum()
        # self.weights = {1: pos_weight, 0: neg_weight}
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        fname = self.dataset.iloc[idx]["Path"]
        blob_client = self.blob_service_client.get_blob_client(container="echo-data-lake", blob=fname)
        download_stream = blob_client.download_blob()
        download_arr = np.frombuffer(download_stream.readall(), dtype=np.float64)
        data = np.array(download_arr).reshape(164, 164, 164)
        data[data < -1000] = -1000
        data[data > 1000] = 1000

        #!
        data = self.transform(**{"image": data})["image"]
        data = torch.FloatTensor(data)
        if "labels" in self.dataset.columns:
            label = torch.tensor(self.dataset.iloc[idx]["labels"]).float()
        else:
            label = -1
        # label = torch.tensor(self.dataset.iloc[idx]['ECHO_lvef_value'] < 40).float()
        data = data.permute((2, 0, 1))
        data = data.unsqueeze(0)

        if "ECHO_lvef_value" in self.dataset.columns:
            lvef = self.dataset.iloc[idx]["ECHO_lvef_value"]
        else:
            lvef = -1

        return data, label, 0, lvef / 100, idx

    def stratify(self, task_str=None, time_str=None):
        if task_str is not None:
            stratified_val = self.dataset.query(task_str)
        else:
            stratified_val = self.dataset.query("index==index")  #! return all
        if time_str is not None:
            assert time_str in ["1d", "1w", "1m"]
            time_arg1, time_arg2 = int(time_str[0]), time_str[1].upper()
            stratified_val = stratified_val[
                (stratified_val["ECHO_study_dttm_y"] - stratified_val["CT_StudyDttm_y"])
                <= np.timedelta64(time_arg1, time_arg2)
            ]
        stratified_index = (self.dataset["Path"].isin(stratified_val["Path"])).to_numpy().nonzero()[0]
        return stratified_index


class CTLoader:
    def __init__(self, base_dir, blob_service_client, batch_size, num_workers) -> None:
        self.train_transform = Compose(
            [RandomCrop((144, 144, 164), always_apply=True, p=1.0), Flip(0, p=0.5)],
            p=1.0,
        )
        self.test_transform = Compose([CenterCrop((144, 144, 164), always_apply=True, p=1)], p=1.0)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_dir = base_dir
        self.blob_service_client = blob_service_client

        self.file_mapping = {
            "train": "298_Preprocessed_CT_ECHO_columbia_train.csv",
            "val": "298_Preprocessed_CT_ECHO_columbia_val.csv",
            "ctest": "298_Preprocessed_CT_ECHO_columbia_test.csv",
            "wtest": "298_Preprocessed_CT_ECHO_cornell_smooth.csv",
            "wprospect": "cornell_prospective3.csv",
            "cprospect": "columbia_prospective3.csv",
            "wproblem": "cornell_problematic.csv",
            "cproblem": "columbia_problematic.csv",
            "opportun": "opportunistic_screening_final.csv",
        }

    def run(self, mode):
        files = pd.read_csv(os.path.join(self.base_dir, self.file_mapping[mode]))
        contrast = pd.read_csv(os.path.join(self.base_dir, "contrast_issue.csv"))
        amyloids = pd.read_csv(os.path.join(self.base_dir, "Selected_CT_ECHO_PYP_16sept.csv"))
        pyp = pd.read_csv(os.path.join(self.base_dir, "Selected_CT_ECG_ECHO_PYP_all.csv"))
        #! Remove contrast
        # files = files[~files["Path"].isin(contrast["Path"].to_list())]

        #! Remove PYP?
        # files = files[~files["Study Instance UID"].isin(pyp["Study Instance UID_x"].unique())]
        #! Male or Female in train
        # if mode == 'train':
        # files = files.query('Patients_Sex == "M" ')
        # files = files.query('Patients_Sex == "F" ')
        #! Remove amyloids
        files = files[~files["Study Instance UID"].isin(amyloids["Study Instance UID"].to_list())]

        # dataset = CTDataset(files, self.train_transform, self.blob_service_client)

        if mode == "train":
            # files = pd.concat([files, pd.read_csv(os.path.join(self.base_dir, "columbia_prospective_additional.csv"))])
            dataset = CTDataset(files, self.train_transform, self.blob_service_client)
            loader = DataLoader(
                dataset,
                self.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=False,
                num_workers=self.num_workers,
            )
        elif mode == "val" or mode == "ctest" or mode == "wtest":
            dataset = CTDataset(files, self.test_transform, self.blob_service_client)
            loader = DataLoader(
                dataset,
                self.batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                num_workers=self.num_workers,
            )
        return loader, files

    def run_eval(self, mode):
        files = pd.read_csv(os.path.join(self.base_dir, self.file_mapping[mode]))

        dataset = CTDataset(files, self.test_transform, self.blob_service_client)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
        )
        return loader, files
