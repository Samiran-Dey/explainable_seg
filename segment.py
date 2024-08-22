
import os
import numpy as np
import cv2
import copy
import math
import SimpleITK as sitk
import itk
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology, segmentation
from sklearn.metrics import r2_score
import pickle
from tqdm import tqdm

save_path = 'Code/checkpoints/ricord/'
if not os.path.exists(save_path):
  os.makedirs(save_path)

# Load data
print('Loading data ...')
img_path = 'Dataset/Lung_mask/ricord_1a_mask'
with open(img_path, "rb") as fp: 
  data_paths, patient_dicom, patient_pixels, lung_mask = pickle.load(fp)



# Lung segmentation

## Obtain percentage label

print('Obtaining percentage label ...')

def assign_percentage(msk):
    i, start, end=0,0,0
    while(end <= len(msk)//2):
        while(i<len(msk) and (not np.any(msk[i]))):
            i+=1
        start=i
        while(i<len(msk) and np.any(msk[i])):
            i+=1
        end=i-1
    no_of_slices=end-start+1
    interval=100/no_of_slices
    lb=-(interval*start)
    ub=100+(interval*(len(msk)-(end+1)))
    prec=np.arange(lb,ub,interval)
    return prec

perc_label=[]
for msk in lung_mask:
    p=assign_percentage(msk)
    perc_label.append(p)

def view_percentage(p_label, percentage):
    #perc_label=assign_percentage(ct_image)
    idx = (np.abs(p_label - percentage)).argmin()
    return idx

## Pleura Removal
print('Generate pleura mask ...')

def extract_lungs(dcm_pixels, mask):
    copied_pixels = copy.deepcopy(dcm_pixels)
    for i, msk in enumerate(mask):
        get_high_vals = msk == 0
        copied_pixels[i][get_high_vals] = 0
    return copied_pixels

def remove_pleura(lung_masks):
  dilation_mask=[]
  kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  for slice_no in range(len(lung_masks)):
    masks=np.array(lung_masks[slice_no])
    masks[masks!=0]=255
    masks=cv2.erode(masks, kernel, iterations=3)
    masks=extract_lungs(lung_masks[slice_no],masks)
    dilation_mask.append(masks)
  return np.array(dilation_mask)

## Remove Organ Lining
print('Remove organ lining ...')

def check_lung(organ, msk):
    msk[organ==0]=0
    val=np.unique(msk)
    check_val=np.zeros(6,int)
    for v in val:
        check_val[v]=1
    if check_val[1] or check_val[2]:
        return True
    else:
        return False

def segment_lr_lung(msk):
  ll=np.array(msk)
  ll[(ll==1) | (ll==2)]=0
  rl=np.array(msk)
  rl[(rl==3) | (rl==4) | (rl==5)]=0
  return ll,rl

# return lung edge after removing perc percent edge from below
def get_edge(msk, perc=0.33):
  msk=cv2.Canny(msk,1,1,3)
  if np.any(msk):
    nz=np.nonzero(msk)
    height=nz[0][len(nz[0])-1]-nz[0][0]
    height=nz[0][len(nz[0])-1]-int(height*perc)
    msk[height:,0:]=0
  return msk

def remove_upwards(scan, no, thickness, msk):
  th=thickness
  kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

  while th<4:
    sl=scan[no]
    sl[scan[no+1]==0]=0

    sl[msk!=0]=0

    edge=np.array(scan[no])
    edge[edge!=0]=255
    edge=get_edge(edge,0.6)
    edge[edge!=0]=1
    edge=morphology.binary_dilation(edge, np.ones((3,3),int))
    sl[edge==1]=0

    mask=cv2.erode(msk, kernel, iterations=math.floor(thickness))

    th+=thickness

    no-=1

def remove_downwards(scan, no, thickness, p_label, msk):
  th=0
  kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
  msk=cv2.dilate(msk, kernel, iterations=10)
  while p_label[no]<87 and no<len(scan)-1:
    sl=scan[no]
    sl[scan[no+1]==0]=0
    if th<=12:
        sl[msk!=0]=0

        edge=np.array(scan[no])
        edge[edge!=0]=255
        edge=get_edge(edge,0.55)
        edge[edge!=0]=1
        edge=morphology.binary_dilation(edge, np.ones((3,3),int))
        sl[edge==1]=0

        msk=cv2.dilate(msk, kernel, iterations=math.ceil(thickness)+1)

    th+=thickness
    no+=1
  return no

def remove_last(scan, no):
  for scan_no in range(no, len(scan)-1):
    sl=scan[scan_no]

    edge=np.array(sl)
    sl[scan[scan_no+1]==0]=0
    edge[edge!=0]=255
    edge=get_edge(edge, 0.6)
    edge[edge!=0]=1
    edge=morphology.binary_dilation(edge, np.ones((3,3),int))
    sl[edge==1]=0
    msk1=np.roll(sl,7,axis=0)
    sl[msk1==0]=0

def remove_other_organ(organ, lung_msk, no, thickness):
    msk=np.array(lung_msk[no])
    lung=check_lung(organ, msk)
    th=0
    area=[]
    slice_no=[]
    copy_msk=[]
    while th<=15 and no<len(lung_msk):
        msk=np.array(lung_msk[no])
        ll,rl=segment_lr_lung(msk)
        edge=rl
        if lung:
            edge=ll
        edge[edge!=0]=1
        area.append(np.sum(edge))
        slice_no.append(no)
        copy_msk.append(np.array(edge))
        edge[edge!=0]=255
        edge=get_edge(edge, 0.4)
        edge[edge!=0]=1
        edge=morphology.binary_dilation(edge, np.ones((12,12),int))
        msk=lung_msk[no]
        msk[edge==1]=0
        th+=thickness
        if lung:
            no-=1
        else:
            no+=1
    if no>=len(lung_msk):
        return
    diff_area=[]
    for i in range(1,len(area)):
        diff=int(area[i-1])-int(area[i])
        diff_area.append(abs(diff))
    ind=np.argmax(diff_area)
    th=0
    if slice_no[0]<slice_no[1]:
        in1=ind+1
    else:
        in1=ind
    in1+=1
    while 0<=ind<len(area) and area[ind]>50 and th<4:
        if slice_no[0]<slice_no[1]:
            no=slice_no[ind]
        else:
            no=slice_no[ind]-1
        no+=1
        msk=lung_msk[no]
        edge=np.array(copy_msk[in1-1])
        msk_base=lung_msk[no+1]
        msk[msk_base==0]=0
        edge[edge!=0]=255
        edge=get_edge(edge, 0.5)
        edge[edge!=0]=1
        edge=morphology.binary_dilation(edge, np.ones((12,12),int))
        msk=lung_msk[no]
        if slice_no[0]<slice_no[1]:
            msk=lung_msk[no+1]
            ind-=1
        else:
            ind+=1
        msk[edge==1]=0
        th+=thickness
        in1-=1

def remove_organ_lining(ct_scan, thickness, p_label):
    organ=[]
    for scan_no in range(len(ct_scan)):
        dm=np.array(ct_scan[scan_no])
        dm[dm!=0]=1
        dm=1-dm
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dm, connectivity=8)
        if nb_components==3 and p_label[scan_no]>57 and p_label[scan_no]<79:
            label=np.argmin(stats[:,4:].flatten())
            output[output!=label]=0
            output[output!=0]=1
            organ.append(np.uint8(output))
            break

    if len(organ)>0:
        for i in range(0,len(organ)):
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            org=cv2.dilate(organ[i], kernel, iterations=2)
            if i==0:
                remove_other_organ(org, ct_scan, scan_no, thickness)
            remove_upwards(ct_scan, scan_no-1, thickness, np.array(org))
            scan_no=remove_downwards(ct_scan, scan_no, thickness, p_label, org)
        remove_last(ct_scan, scan_no)
    else:
        ind=view_percentage(p_label, 68)
        remove_last(ct_scan, ind)

## Extract Lungs
print('Extract lungs ...')

seg_lung_pixels=[]
pleura_mask=[]
organ_mask=[]
for scan_no in tqdm(range(len(patient_pixels))):
    pleura_mask.append(remove_pleura(lung_mask[scan_no]))
    organ_mask.append(np.array(pleura_mask[scan_no]))
    remove_organ_lining(organ_mask[scan_no], patient_dicom[scan_no][0].SliceThickness, perc_label[scan_no])
    seg_lung_pixels.append(extract_lungs(patient_pixels[scan_no], organ_mask[scan_no]))


## Remove Vessels
print('Remove vessels ...')

def segment_vessels(ct_slice):
  input_image=itk.GetImageViewFromArray(ct_slice)
  ImageType = type(input_image)
  Dimension = input_image.GetImageDimension()
  HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
  HessianImageType = itk.Image[HessianPixelType, Dimension]

  objectness_filter = itk.HessianToObjectnessMeasureImageFilter[
    HessianImageType, ImageType
  ].New()
  objectness_filter.SetBrightObject(False)
  objectness_filter.SetScaleObjectnessMeasure(False)
  objectness_filter.SetAlpha(0.5)
  objectness_filter.SetBeta(1.0)
  objectness_filter.SetGamma(5.0)

  multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[
    ImageType, HessianImageType, ImageType
  ].New()
  multi_scale_filter.SetInput(input_image)
  multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
  multi_scale_filter.SetSigmaStepMethodToLogarithmic()

  OutputPixelType = itk.UC
  OutputImageType = itk.Image[OutputPixelType, Dimension]

  rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
  rescale_filter.SetInput(multi_scale_filter)

  ct_img=rescale_filter.GetOutput()
  vessels_lung = itk.GetArrayViewFromImage(ct_img)
  return np.array(vessels_lung)

def extract_vessels_mask(ct_vol, lungs_mask):
  vessels_mask=[]
  for slice_no in range(len(ct_vol)):
    vessel_mask=segment_vessels(ct_vol[slice_no])
    vessel_mask[vessel_mask<=10]=0
    vessel_mask[vessel_mask!=0]=1
    msk=np.array(lungs_mask[slice_no])
    bg=np.zeros(msk.shape,int)
    bg[(vessel_mask==0) & (msk!=0)]=1
    processed1= morphology.remove_small_objects(bg.astype(bool), min_size=60, connectivity=1).astype(int)
    bg=morphology.binary_dilation(processed1, np.ones((2,2),int))
    processed2= morphology.remove_small_objects(bg.astype(bool), min_size=460, connectivity=1).astype(int)
    bg=bg+morphology.binary_dilation(processed2)
    vessel_mask[bg]=0
    vessel_mask[vessel_mask!=0]=msk[vessel_mask!=0]
    vessels_mask.append(vessel_mask)

  return vessels_mask

vessel_mask=[]
lung_pixels=[]
for scan_no in tqdm(range(len(seg_lung_pixels))):
  vessel_mask.append(extract_vessels_mask(seg_lung_pixels[scan_no], np.array(organ_mask[scan_no])))
  lung_pixels.append(extract_lungs(patient_pixels[scan_no], vessel_mask[scan_no]))


## Removal of motion artifacts
print('Remove motion artefacts ...')

def get_art_mask(masks, p_label):
  art_mask=[]
  for slice_no in range(len(masks)):
    msk=masks[slice_no]
    img=np.array(msk)
    if p_label[slice_no]<7:   #for first 7 percent slices don't remove edge from edge mask
        img[img!=0]=255
        img=get_edge(img,0)
        img[img!=0]=1
        img=morphology.binary_dilation(img, np.ones((10, 10),int))
    else:
        ll,rl=segment_lr_lung(msk)
        ll[ll!=0]=255
        rl[rl!=0]=255
        ll_img=get_edge(ll, 0.34)
        img=get_edge(rl)
        img[(ll_img!=0) | (img!=0)]=1
        if p_label[slice_no]>15:
          img=morphology.binary_dilation(img, np.ones((10,10),int))
        else:
          img=morphology.binary_dilation(img, np.ones((7,7),int))
    art_mask.append(img)
  return art_mask

artifact_mask=[]
for scan_no in tqdm(range(len(seg_lung_pixels))):
  artifact_mask.append(get_art_mask(pleura_mask[scan_no], perc_label[scan_no]))


# Define infection region using HU
print('Segmenting ...')

def extract_edema_mask(ct_img):
  img=np.array(ct_img)
  img[img<-700]=0
  img[img>30]=0  #21
  img[img!=0]=1
  return img

def remove_artifacts(ed_mask, art_mask):
  nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(ed_mask), connectivity=8)
  nz=np.nonzero(art_mask)
  nz=np.stack((nz[1],nz[0]),axis=1)
  centroids=np.rint(centroids)
  for cc in range(1,nb_components):
    if stats[cc][4]<46:
        ed_mask[output==cc]=0
        continue
    pos=np.where((nz==centroids[cc]).all(axis=1))
    if (len(pos[0])!=0) and (stats[cc][4]<(len(nz)//2)):
        ed_mask[output==cc]=0
  return ed_mask

def remove_noncontinuous(ed_mask):
    for slice_no in range(1,len(ed_mask)-1):
        if (not np.any(ed_mask[slice_no-1])) and (not np.any(ed_mask[slice_no+1])):
            em=ed_mask[slice_no]
            em[em!=0]=0

def refine_edema_mask(ed_mask, art_mask):
    img=[]
    for slice_no in range(len(ed_mask)):

        ct=ed_mask[slice_no]
        im= morphology.remove_small_objects(ct.astype(bool), 30).astype(int)
        mask_x, mask_y = np.where(im== 0)
        ct[mask_x, mask_y] = 0
        ct=remove_artifacts(ct,art_mask[slice_no])

        binary_ct = ct.astype(bool)
        binary_filled = morphology.remove_small_holes(binary_ct, 100) #55
        ct = segmentation.watershed(binary_filled, ct, mask=binary_filled)

        img.append(ct)
    remove_noncontinuous(img)
    return img

edema_mask_HU=[]
for scan_no in tqdm(range(len(lung_pixels))):
    edema_mask_HU.append(refine_edema_mask(extract_edema_mask(lung_pixels[scan_no]),artifact_mask[scan_no]))

# save data
print('Saving data ...')
images = [perc_label, pleura_mask, organ_mask, seg_lung_pixels, vessel_mask, lung_pixels, artifact_mask, edema_mask_HU]
with open(save_path, "wb") as fp:   
  pickle.dump(images, fp)

print('Data saved ...')


