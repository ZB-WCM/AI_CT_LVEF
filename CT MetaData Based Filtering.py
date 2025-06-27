#!/usr/bin/env python
# coding: utf-8
# %%


import pydicom
import pandas as pd
from azureml.fsspec import AzureMachineLearningFileSystem
from io import BytesIO
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import tqdm


pd.set_option('display.max_columns', 500)

pd.set_option('display.max_columns', 500)

from IPython.display import display, HTML
display(HTML("<style>div.output_scroll { height: 44em; }</style>"))

import ast


# # Loading .dcm files from location


# %%


# Keep datastore name updated
#subscription   = 
#resource_group = 
#workspace      = 
#datastore_name = 

# long-form datastore URI format
base_uri = f'azureml://subscriptions/{subscription}/resourcegroups/{resource_group}/workspaces/{workspace}/datastores/{datastore_name}/paths/'
path_uri = base_uri + ''
print(path_uri)


# %%


k = list(range(0, 5, 1))
print(k)



# %%


df_file_ct=pd.DataFrame()
for i in tqdm(k, desc="Block Loading"):

    # Create a blob client using the blob file name as the name for the blob
    blob_file_name = f"" # <-- Change to your ID at end of filename
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_file_name)
    print(blob_file_name)
    # Download the blob array file into memory (but not to local storage)
    download_stream = blob_client.download_blob()
    df_file = pd.read_json(download_stream.readall())
    df_file_ct = df_file_ct.append(df_file, ignore_index=True)


# %%


print(len(df_file_ct))


# %%


column_names = df_file_ct.columns.tolist()
print("Column Names:", column_names)


# %%


patients = list(set(df_file_ct['EMPI'].values))
print(len(patients))


# %%


print(df_file_ct["Manufacturer"].unique())

df_file_ct = df_file_ct[df_file_ct['Manufacturer'] == 'Canon Medical Systems']
# %%


patients = list(set(df_file_ct["Study Instance UID"].values))
print(len(patients))


# %%


# Show all possible "Image Type" and counts
# group filtered df and take first slice in every series
df_file_ct_first = df_file_ct.groupby('Series Instance UID').first()
settemp = set(df_file_ct_first['Image Type'])
#val, count = np.unique(df_file_ct_first['Image Type'].to_numpy(), return_counts=True)
#list(zip(val, count))
df_file_ct_series = df_file_ct.groupby('Series Instance UID')
#print(len(np.unique(df_file_ct['Series Instance UID'])))
#print(len(df_file_ct))


# %%


#The number of slices within a qualified series should be larger than 10
df_filtered = df_file_ct[df_file_ct_series['Series Instance UID'].transform('count') >= 10]
print(len(df_filtered))
df_filtered.columns


# %%


image_types_exclude = ["SECONDARY", "LOCALIZER", "MPR", "CT_SOM5 MPR"]
# SECONDARY: Usually Screen Shots
# LOCALIZER: Scout Images
# MPR:
image_types = list(set(df_filtered['Image Type'].values.tolist()))
image_types


# %%


#Filtering according to Image Type, Slice Thickness
print(f"number of series before filtering: {len(df_filtered)}")
df_filtered_ExK = df_filtered[~df_filtered['Image Type'].fillna("").str.contains('|'.join(image_types_exclude))]
#df_filtered_ExK=df_filtered
df_filtered_ExK_Axial = df_filtered_ExK[df_filtered_ExK["Series Description"].str.contains("axial", case=False) | df_filtered_ExK["Image Type"].str.contains("axial", case=False)]
df_filtered_ExK_Axial_filtered_first = df_filtered_ExK_Axial.groupby('Series Instance UID').first()
print(f"number of series after filtering of Image type and series description: {len(df_filtered_ExK_Axial_filtered_first)}")
#Slice Thickness: Statistics and Visualization before filtering
df_filtered_first_prior=df_filtered_ExK_Axial
df_filtered_ExK_Axial_ST = df_filtered_ExK_Axial[df_filtered_ExK_Axial["Slice Thickness"].str.replace(",", "").astype(float) <= 2]
# group filtered df and take first slice in every series
df_filtered_firstT = df_filtered_ExK_Axial_ST.groupby('Series Instance UID').first()
df_filtered_first=df_filtered_ExK_Axial_ST
print(f"number of series after filtering accoring to slice thickness: {len(df_filtered_firstT)}")
val, count = np.unique(df_filtered_first['Image Type'].to_numpy(), return_counts=True)
tmplist = list(zip(val, count))


# %%


unique_PatIDs = list(set(df_filtered_first["EMPI"].values))
print(len(unique_PatIDs))
df_filtered_first.columns


# %%


unique_body_part = list(set(df_filtered_first["Body Part Examined"].values))
print(unique_body_part)
df_filtered_first.columns


# %%


unique_body_part = list(set(df_filtered_first["Study Description"].values))
print(unique_body_part)


# %%


#Body Part Examined Filtering
image_types_exclude = ["ABDOMEN"]
df_filtered_first1 = df_filtered_first[~df_filtered_first['Body Part Examined'].fillna("").str.contains('|'.join(image_types_exclude))]
#df_filtered_first1=df_filtered_first
print(f"number of series before filtering: {len(df_filtered_first)}")
print(f"number of series after filtering: {len(df_filtered_first1)}")


# %%


#Modality Based Filtering
unique_modality = list(set(df_filtered_first1["Modality"].values))
print(unique_modality)
df_filtered_first1.columns


# %%


df_filtered_first2 = df_filtered_first1[df_filtered_first1["Modality"].str.contains("CT", case=False)]
print(f"number of series before filtering: {len(df_filtered_first1)}")
print(f"number of series after filtering: {len(df_filtered_first2)}")


# %%


#Study Description Based Filtering
unique_study = list(set(df_filtered_first2["Study Description"].values))
print(unique_study)
df_filtered_first2.columns


# %%


df_filtered_first3 = df_filtered_first2[df_filtered_first2["Study Description"].str.contains("CT CHEST WITHOUT IV CONTRAST", case=False) | df_filtered_first2["Study Description"].str.contains("WITHOUT INTRAVENOUS", case=False) | df_filtered_first2["Study Description"].str.contains("WO CONTRAST", case=False)]
print(f"number of series before filtering: {len(df_filtered_first2)}")
print(f"number of series after filtering: {len(df_filtered_first3)}")


# %%


unique_study = list(set(df_filtered_first3["Study Description"].values))
print(unique_study)
df_filtered_first3.columns


# %%


unique_kernel = list(set(df_filtered_first3["Convolution Kernel"].values))
print(unique_kernel)
df_filtered_first3.columns


# %%


#Filtering using Standard Convolutional Kernel
#df_filtered_first4=df_filtered_first3
image_types_exclude = [ "CARDIAC","YC", "YA", "FC30", "FC51", "FC56", "FC81", "FC86", "B60f", "B70f", "B90s", "Br58f", "Br59f", "Br64f", "Br59d", "Br69d", "BI57f", "BI57d", "I70f", "Lung", "LUNG", "HD LUNG",'BONEPLUS', 'BONE']
df_filtered_first4 = df_filtered_first3[~df_filtered_first['Convolution Kernel'].fillna("").str.contains('|'.join(image_types_exclude))]
print(f"number of series before filtering: {len(df_filtered_first3)}")
print(f"number of series after filtering: {len(df_filtered_first4)}")


# %%


unique_PatIDs = list(set(df_filtered_first4["EMPI"].values))
print(len(unique_PatIDs))


# %%


unique_study = list(set(df_filtered_first4["Study Instance UID"].values))
print(len(unique_study))


# %%


unique_filename = list(set(df_filtered_first4["Filename"].values))
print(len(unique_filename))


# %%


unique_Series_Instance_UID = list(set(df_filtered_first4["Series Instance UID"].values))
print(len(unique_Series_Instance_UID))
#print(unique_Series_Instance_UID)


# %%


sex=df_filtered_first4["Patient's Sex"].values
print(sex)


# %%


age=df_filtered_first4["Patient's Age"].values
print(age)
# # Save File as CSV


# %%


df_filtered_first4['Instance Number'].unique()


# %%


print(df_filtered_first4["Manufacturer"].unique())


# %%


#Pre-processing
df_filtered = df_filtered_first4
file=[] 
pati=[]
si=[]
su=[]
sex=[]
age=[]
i=[]
for patid in tqdm(unique_PatIDs):
    study = list(set(df_filtered.loc[df_filtered["EMPI"]==patid]["Study Instance UID"].values))
    s=0
    a=0
    for stu in study:
        series = list(set(df_filtered.loc[df_filtered["Study Instance UID"]==stu]["Series Instance UID"].values))
        min_thickness = float('inf')  
        min_series_name = None
        for sid in series:
            SliceThickness = df_filtered.loc[df_filtered["Series Instance UID"] == sid, "Slice Thickness"].values[0]
            s=df_filtered.loc[df_filtered["Series Instance UID"] == sid, "Patient's Sex"].values[0]
            a=df_filtered.loc[df_filtered["Series Instance UID"] == sid, "Patient's Age"].values[0]
            SliceThickness = float(SliceThickness)
            if SliceThickness < min_thickness:
                min_thickness = SliceThickness
                min_series_name = sid
        filenames = list(df_filtered.loc[df_filtered['Series Instance UID']==min_series_name]['Filename'].values)
        for f in filenames:
            instance=df_filtered.loc[df_filtered["Filename"] == f, "Instance Number"].values[0]
            i.append(instance)
            file.append(f)
            pati.append(patid)
            si.append(min_series_name)
            su.append(stu)
            sex.append(s)
            age.append(a)


# %%


print(len(file))
print(len(pati))
print(len(si))
print(len(su))
print(len(sex))
print(len(age))
df = pd.DataFrame(list(zip(pati,su,si,file,i)), columns=['EMPI','Study Instance UID','Series Instance UID','Filenames','Instance Number'])
print(df.head())
df.shape

