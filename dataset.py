import os
import numpy as np
import cv2
import pydicom
from glob import glob
from lungmask import mask
import SimpleITK as sitk
import pylibjpeg
import gdcm
import torch
import pickle

# set paths
paths='Dataset/MIDRC_RICORD_1A/'
save_path = 'Dataset/Lung_mask/ricord_1a_mask'

# function to load ct scans by reading dicom files
def load_scan(paths):
    slices = [pydicom.read_file(path ) for path in paths]
    
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
            
    return slices

#function to convert to HU scale
def convert_HU(scans):
    new_pix=[]
    for ct in scans:
        ct.PhotometricInterpretation = 'YBR_FULL'
        img=ct.pixel_array
        rescaled_img=img*ct.RescaleSlope+ct.RescaleIntercept
        if rescaled_img.shape != (512,512):
          rescaled_img=cv2.resize(rescaled_img, (512, 512), interpolation = cv2.INTER_CUBIC)
        new_pix.append(rescaled_img)
    return np.array(new_pix)

#function to obtain lung mask
def get_mask(ct_filename, type='R231'):
    img = sitk.ReadImage(ct_filename)
    img1 = np.array(img)
    if type=='R231':
        mask_out = mask.apply(img)
    elif type=='LTRCLobes':
        model = mask.get_model('unet','LTRCLobes')
        mask_out = mask.apply(img, model)
    else:
        mask_out = mask.apply_fused(img)
    try:
        mask_out = np.array(mask_out).reshape(patient_pixels[0][0].shape)
    
    except:
        mask_out = np.array(mask_out).reshape(img1.shape)
        mask_out = cv2.resize(mask_out, (512, 512), interpolation = cv2.INTER_CUBIC)
        
    return mask_out

dirs = os.listdir(paths)
dirs.sort()
dirs = dirs[:50]
data_paths=[]

for folder in dirs:
    patient_folder=paths+folder+'/'
    folders = os.listdir(patient_folder)
    folders.sort()
    for fold in folders:
      patient_folder=patient_folder+fold+'/'
      path = glob(patient_folder + '*.dcm')
      data_paths.append(sorted(path))
      print (folder,f' contains total of {len(path)} DICOM images.' )

# read data
print('Loading Images ...')
patient_dicom=[]
patient_pixels=[]
for path in data_paths:
    dicom=load_scan(path)
    patient_dicom.append(dicom)
    patient_pixels.append(convert_HU(dicom))

# extract mask
print('Extracting masks ...')
lung_mask=[]
for path in data_paths:
  msk_LTRCLobes_R231=[]
  for slice_no in range(len(path)):
      msk_LTRCLobes_R231.append(get_mask(path[slice_no], type='LTRCLobes_R231'))
  lung_mask.append(msk_LTRCLobes_R231)

# save the data
print('Saving data ...')
images = [data_paths, patient_dicom, patient_pixels, lung_mask]
with open(save_path, "wb") as fp: 
  pickle.dump(images, fp)



