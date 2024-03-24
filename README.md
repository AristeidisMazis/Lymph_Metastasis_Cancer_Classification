# Lymph Metastasis Cancer Classification using Deep Learning


### Using the power of Deep Learning to identify and locate metastatic cancer of the lymphatic system of breast cancer patients in WSI(Whole Slide Image). Created to aid the pathology community

* WSI Dataset - http://gigadb.org/dataset/100439


## WSI_preprocessing.py and WSI_map_data.py

#### Extracting the dataset and necessary files for our Deep Learning model

* Import necessary libraries
* Create WSI binary masks to save processing power, time and space
* Extract patches

#### After extraction make sure to create evenly distributed normal and tumor folders in order to avoid overfitting

![tain method](https://github.com/AristeidisMazis/Lymph_Metastasis_Cancer_Classification/assets/164747509/068e8649-7894-4b81-8fde-fffc538b61dc)

#### *Binary masks should look like this

![Untitled-1](https://github.com/AristeidisMazis/Lymph_Metastasis_Cancer_Classification/assets/164747509/a158e57c-bd77-41aa-a28b-26c7dda5e94d)![Untitled-3](https://github.com/AristeidisMazis/Lymph_Metastasis_Cancer_Classification/assets/164747509/d1d3516b-cc7f-482f-b5af-2c55351a23aa)

## Deep Learning model

#### The model was implemented in Google Colab using TensorFlow and Keras libraries. EfficientNetB0 was used as pretrained model 

#### Results:

##### For 256x256 patches:

Accuracy: 95.09%

##### For 512x512 patches:

Accuracy: 96.97%

## Heatmap

### A visual illustration to locate cancer in the WSI

![tumor_104_bounding](https://github.com/AristeidisMazis/Lymph_Metastasis_Cancer_Classification/assets/164747509/b1d2e89f-91cd-43a8-bacd-c0ca4c0b9549)![image](https://github.com/AristeidisMazis/Lymph_Metastasis_Cancer_Classification/assets/164747509/40d1f50a-ad3a-428c-b8d1-85003a9bca1b)

#### Cancer is located correctly, although there are some false-possitive spots. 

## Improvement methods

* More data along with a better deep learning model
* Better pre-processing, targeted in difficult-to-evaluate spots
* Script to automate binary mask extraction using machine learning
* Better fine tuning

### Usefull links
https://openslide.org/

