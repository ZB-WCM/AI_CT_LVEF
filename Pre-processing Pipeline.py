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
import os
import ast
from pydicom.data import get_testdata_file
from pydicom.uid import ImplicitVRLittleEndian


# %%


#Datastorage
#subscription   = 
#resource_group = 
#workspace      = 
#datastore_name = 

base_uri = f'azureml://subscriptions/{subscription}/resourcegroups/{resource_group}/workspaces/{workspace}/datastores/{datastore_name}/paths/'
path_uri = base_uri + 'dicoms/'
print(path_uri)


# %%


#transform to HU, Windowing and Resampling

def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image

def resample_m(image, voxel_spacing, target_voxel_spacing = [2,2,2], target_grid_size=[164,164,164]):
    resize_factor = voxel_spacing/target_voxel_spacing
    resampled_image = ndimage.interpolation.zoom(image, resize_factor)
    return resampled_image


# %%


# Displace single slice
def display_slice(filename):
    with AzureMachineLearningFileSystem(filename).open() as fp:
            dcm_ct = pydicom.read_file(fp)
            image_array = dcm_ct.pixel_array
            hu_image = transform_to_hu(dcm_ct, image_array)
            chest_image = window_image(hu_image, 0, 2000)
            chest_image=add_pad(chest_image)
            xy_axis = np.arange(dcm_ct.Columns)*dcm_ct.PixelSpacing[0]
            plt.figure(figsize=[14,6])
            plt.imshow(chest_image, cmap=plt.cm.gray)
            plt.colorbar() 
            plt.show()
    return

# Emsemble slices into series
def series_ensemble_prior(filenames, rows, columns):
    voxel_array = np.empty([len(filenames), rows, columns])
    window_info = np.empty([len(filenames), 2])
    for i,fn in enumerate(filenames):
        with AzureMachineLearningFileSystem(base_uri + fn).open() as fp:
            dcm_ct = pydicom.read_file(fp)
            image_array = dcm_ct.pixel_array
            img_rescale = image_array*dcm_ct.RescaleSlope + dcm_ct.RescaleIntercept
            window_center = dcm_ct.WindowCenter
            window_width  = dcm_ct.WindowWidth
            if isinstance(window_center, pydicom.multival.MultiValue) > 0:
                window_center = window_center[0]
            if isinstance(window_width, pydicom.multival.MultiValue) > 0:
                window_width = window_width[0]
            window_lower = window_center - window_width/2
            window_upper = window_center + window_width/2
            voxel_array[i,...] = img_rescale
            window_info[i,...] = [window_lower, window_upper]
            fp.close()
    print(f"First window info: window_lower {window_info[0][0]}, window_upper {window_info[0][1]}")
    return voxel_array, window_info

# Emsemble slices into series
def series_ensemble_preprocess(filenames,sid, rows, columns):
    voxel_array = np.empty([len(filenames), rows, columns])
    window_info = np.empty([len(filenames), 2])
    for i,fn in enumerate(filenames):
        with AzureMachineLearningFileSystem(path_uri + fn).open() as fp:
            dcm_ct = pydicom.read_file(fp)
            if dcm_ct.pixel_array is not None:
                image_array = dcm_ct.pixel_array
            else:
                print(f"Delete file {sid} as it has a None pixel_array.")
            window_center = 0
            window_width  = 2000
            hu_image = transform_to_hu(dcm_ct, image_array)
            window_image1=window_image(hu_image,0,2000)
            voxel_array[i,...] =  window_image1
            window_info[i,...] = [-1000, 1000]
            fp.close()
    return voxel_array, window_info,dcm_ct.SliceThickness,dcm_ct.PixelSpacing

# Display series as a sequence
def display_series(voxel_array, window_info):
    for ind_slice in range(voxel_array.shape[0]):
        display.clear_output(wait=True)
        time.sleep(.03)
        img_rescale = voxel_array[ind_slice,...]
        [window_lower, window_upper] = window_info[ind_slice,...]
        plt.figure(figsize=[18,9])
        plt.imshow(img_rescale, cmap=plt.cm.gray)
        plt.clim([window_lower, window_upper]) 
        plt.title(f'Slice {ind_slice+1} / {voxel_array.shape[0]}')
        plt.show()
    return

# Display series side view
def display_series_size_view(voxel_array, window_info, slice_thickness, x_spacing):
    [window_lower, window_upper] = window_info[0,...]
    img_rescale = voxel_array[:,:,voxel_array.shape[2]//2]
        
    plt.figure(figsize=[18,9])
    plt.imshow(img_rescale, cmap=plt.cm.gray)
    plt.clim([window_lower, window_upper])
    plt.title(f'Side view at Slice Thickness: {slice_thickness}, Pixel Spacing: {x_spacing}')
    plt.show()
    return

# Display series in a row
def display_series_row(voxel_array, window_info):
    fig, axs = plt.subplots(1,voxel_array.shape[0], figsize=[14,6])
    slice_titles = ['First', 'Middle', 'Last']
    for ind_slice in range(voxel_array.shape[0]):

        img_rescale = voxel_array[ind_slice,...]
        [window_lower, window_upper] = window_info[ind_slice,...]
        axs[ind_slice].imshow(img_rescale, cmap=plt.cm.gray, clim=[window_lower, window_upper])
        axs[ind_slice].set_title(f'{slice_titles[ind_slice]} slice')
    fig.tight_layout()
    plt.show()
    return

# Show image types
def show_image_types(keyword, image_types):
    if not keyword:
        print(f"All Image Types: \n {image_types}")
        image_types_keyword = image_types
    else:
        image_types_keyword = [x for x in image_types if (isinstance(x, str) and keyword in x)]
        print(f"Image Type containing {keyword}: \n {image_types_keyword}")
    return image_types_keyword


# %%
def pad_array(array, target_shape, pad_value):
    # Create a new array with the target shape, filled with the pad value
    padded_array = np.full(target_shape, pad_value)
    # Determine the slices for each dimension to copy the original array into the new array
    original_slices = tuple(slice(0, min(dim, target_dim)) for dim, target_dim in zip(array.shape, target_shape))
    target_slices = tuple(slice(0, min(dim, target_dim)) for dim, target_dim in zip(array.shape, target_shape))
    # Copy the original array values into the padded array
    padded_array[target_slices] = array[original_slices]
    return padded_array

#Cropping and Padding
def zero_pad_3d(array, target_shape):
    padded_array = np.zeros(target_shape)
    padded_array[:array.shape[0], :array.shape[1], :array.shape[2]] = array
    return padded_array

def center_crop_3d(image, crop_size):
    depth, height, width = image.shape
    start_height = max(0, (height - crop_size[1]) // 2)  # Ensure start_height is not negative
    start_width = max(0, (width - crop_size[0]) // 2)  # Ensure start_width is not negative
    end_height = start_height + crop_size[1]
    end_width = start_width + crop_size[0]
    cropped_image = image[:, start_width:end_width,start_height:end_height]
    return cropped_image

def crop_bottom_slices(image, num_slices_to_keep):
    num_slices, height, width = image.shape
    kept_slices = image[:num_slices_to_keep, :, :]
    return kept_slices

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def center_crop_3d_new(image, crop_size):
    """
    Crops the 3D image to a fixed size in the z-dimension and center crops along x and y dimensions.
    
    Parameters:
    image (numpy array): Input 3D image with shape (depth, height, width)
    crop_size (tuple): Desired crop size in the form (crop_depth, crop_height, crop_width)
    
    Returns:
    numpy array: Cropped 3D image with specified crop_size
    """
    depth, height, width = image.shape
    start_height = max(0, (height - crop_size[1]) // 2)
    start_width = max(0, (width - crop_size[2]) // 2)
    end_height = start_height + crop_size[1]
    end_width = start_width + crop_size[2]
    start_depth = max(0, (depth - crop_size[0]) // 2)
    end_depth = start_depth + crop_size[0]
    cropped_image = image[start_depth:end_depth, start_width:end_width, start_height:end_height]
    
    return cropped_image


# %%


#container_name = 


# %%


df_file_ct=pd.DataFrame()
# Create a blob client using the blob file name as the name for the blob
#blob_file_name = f"" # <-- Change to your ID at end of filename
#blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_file_name)
#print(blob_file_name)
# Download the blob array file into memory (but not to local storage)
#download_stream = blob_client.download_blob()
df_file = pd.read_json(BytesIO(download_stream.readall())) #Read the CSV files conatining DICOM study details (EMPI, Study Instance UID, Series Instance UID and Filenames)
df_file_ct = pd.concat([df_file_ct, df_file])


# %%


column_names = df_file_ct.columns.tolist()
print("Column Names:", column_names)


# %%


unique_study = list(set(df_file_ct["Study Instance UID"].values))
print(len(unique_study))


# %%


unique_filename = list(set(df_file_ct["Filenames"].values))
print(len(unique_filename))


# %%


unique_Series_Instance_UID = list(set(df_file_ct["Series Instance UID"].values))
print(len(unique_Series_Instance_UID))


# %%


def numerical_sort(filename):
    parts = filename.split('.')
    numerical_part = parts[-2]
    return int(numerical_part)


# %%


#Pre-processing
df_filtered = df_file_ct
file=[] 
pati=[]
si=[]
su=[]


# %%


column_names = df_filtered.columns
print(column_names)


# %%


print('Files Loading')


# %%


for stu in tqdm(unique_study):
    series = list(set(df_filtered.loc[df_filtered["Study Instance UID"]==stu]["Series Instance UID"].values)) 
    for sid in series:
        filenames = list(df_filtered.loc[df_filtered['Series Instance UID']==sid]['Filenames'].values)
        filenames = sorted(filenames,key=numerical_sort)
        ft = filenames[0]
        fm = filenames[len(filenames)//2] 
        fb = filenames[-1]
        rows = int(512)
        columns = int(512)
        voxel_array,window_info,SliceThickness,PixelSpacing = series_ensemble_preprocess(filenames,sid,rows,columns)
        print(voxel_array.shape)
        x_pixel = float(PixelSpacing[0])
        y_pixel = float(PixelSpacing[1])
        voxel_spacing = np.array([float(SliceThickness), x_pixel, y_pixel])
        resampled_image = resample_m(voxel_array, voxel_spacing)
        crop_size = (164, 164, 164)
        cropped_volume_x_y = center_crop_3d_new(resampled_image, crop_size)
        cropped_volumeT=np.transpose(cropped_volume_x_y)
        target_shape = (164, 164, 164)
        cropped_volumeT = zero_pad_3d(cropped_volumeT, target_shape)
        array_bytes = cropped_volumeT.tobytes()
        #blob_file_name = f".npy"  #named as series UID and .npy format
        #blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_file_name)
        #print("\nUploading to Azure Storage as blob:\n\t" + blob_file_name)
            # Note: you can overwrite an already existing blob file using overwrite=True
       # blob_client.upload_blob(array_bytes, overwrite=True)
        for f in filenames:
            file.append(f)
            si.append(sid)
            su.append(stu)


# %%


print('Done Saving')


# %%




