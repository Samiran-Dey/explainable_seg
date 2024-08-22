# A fast domain-inspired unsupervised method to compute COVID-19 severity scores from lung CT

The repository contains the official implementation of the following paper. \
\
Title - **A fast domain-inspired unsupervised method to compute COVID-19 severity scores from lung CT** \
Authors - Samiran Dey, Bijon Kundu, Partha Basuchowdhuri, Sanjoy Kumar Saha and Tapabrata Chakraborti \
DOI - 

## Abstract
There has been a deluge of data-driven deep learning approaches to detect COVID-19 from computed tomography (CT) images over the pandemic, most of which use ad-hoc deep learning black boxes of little to no relevance to the actual process clinicians use and hence have not seen translation to real-life practical settings. Radiologists use a clinically established process of estimating the percentage of the affected area of the lung to grade the severity of infection out of a score of 0-25 from lung CT scans. Hence any computer-automated process that has aspirations of being adopted in the clinic to alleviate the workload of radiologists while being trustworthy and safe, needs to follow this clearly defined clinical process religiously. Keeping this in mind, we propose a simple yet effective methodology that uses explainable mechanistic modelling using classical image processing and pattern recognition techniques. The proposed pipeline has no learning element and hence is fast, as well as exactly mimics the clinical process and hence is transparent. We collaborate with an experienced radiologist to enhance an existing benchmark COVID-19 lung CT dataset by adding the grading labels, which is another contribution of this paper, along with the methodology which has a higher potential of becoming a clinical decision support system (CDSS) due to its rapid and explainable nature.
\
<img src="./images/pipeline.PNG">  </img>


# COVID-19 gradation data
The folder ‘Radiologist gradations’ contains the COVID-19 gradations by our experienced radiologist for lung CT scans from the Medical Imaging Data Resource Center - RSNA International COVID-19 Open Radiology Database - Release 1a (MIDRC-RICORD-1a) of The Cancer Imaging Archive (TCIA). There are additional comments on different exceptional patterns observed in the CT scans like cystic broncho-ectatic changes in the right lung with bilateral pleural effusion, only mosaic perfusions in both lungs, only scars in both lower lobes, only mosaic perfusions bilaterally, only nodular and streaky scars, etc. The original MIDRC-RICORD-1a data can be downloaded from the link below. \
[MIDRC-RICORD-1a](https://www.cancerimagingarchive.net/collection/midrc-ricord-1a/)


# Getting started

## Installation
To install all requirements execute the following line.
```bash
pip install -r requirements.txt 
pip install git+https://github.com/JoHof/lungmask
```
And then clone the repository as follows. 
```bash
git clone https://github.com/Samiran-Dey/explainable_seg.git
cd explainable_seg
```

## Dataset Preparation
The file dataset.py helps in preparing the data. The variable ‘paths’ stores the path to the folder containing the dicom files. The variable ‘save_path’ contains the path to save the processed data and lung mask. Set the paths and execute the file. 
```bash
python3 dataset.py
```

## Obtain infection region segmentation mask
The file segment.py helps to obtain and save the infection region segmentation mask using our novel explainable approach. The variable ‘save_path’ contains the path to save the segmentation masks and ‘img_path’ contains the path to the previously saved lung masks. Set the paths and execute the file. 
```bash
python3 segment.py
```

## Compute grades and obtain results
The file compute_grade.py helps to obtain the COVID-19 grades and compute the result. The variable ‘save_path’ contains the path to save the computed grades and ‘grade_path’ contains the path to the ground truth grades. Set the paths and execute the file. 
```bash
python3 compute_grade.py
```

# Acknowledgement 
[UNet_R231_LTRCLobes](https://github.com/JoHof/lungmask/tree/master)


# Citation
```bash
Dey, S., Kundu, B., Basuchowdhuri, P., Saha, S.K., Chakraborti, T. (2024). A fast domain-inspired unsupervised method to compute COVID-19 severity scores from lung CT. In: International Conference on Pattern Recognition. ICPR 2024.
```

```bash
 
```



