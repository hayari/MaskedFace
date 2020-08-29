# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import cv2
import os

#image_width = 128
#image_height = 128

image_width = 64
image_height = 64

channels = 3
nb_classes = 1

def load_images_from_folder(folder):
    images = []
    
    for subfolder in os.listdir(folder):
      
      for filename in os.listdir(os.path.join(folder,subfolder)):
         
         im = cv2.imread(os.path.join(folder,subfolder,filename))
         im = cv2.resize(im,(image_width,image_height))
         img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
         if img is not None:
            images.append(img)
    return np.array(images)

print("RMFD")
RMFD_masked=load_images_from_folder('./datasets/RMFD/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset')
RMFD_nonmasked=load_images_from_folder('./datasets/RMFD/self-built-masked-face-recognition-dataset/AFDB_face_dataset')


RMFD_masked_labels=np.ones(RMFD_masked.shape[0])
RMFD_nonmasked_labels=np.zeros(RMFD_nonmasked.shape[0])

RFMD_labels=np.hstack((RMFD_masked_labels,RMFD_nonmasked_labels))
RFMD_dataset=np.vstack((RMFD_masked,RMFD_nonmasked))
print("LFW")
LFW_masked=load_images_from_folder('./datasets/lfw_masked/lfw_masked')
LFW_nonmasked=load_images_from_folder('./datasets/lfw')

LFW_masked_labels=np.ones(LFW_masked.shape[0])
LFW_nonmasked_labels=np.zeros(LFW_nonmasked.shape[0])

LFW_labels=np.hstack((LFW_masked_labels,LFW_nonmasked_labels))
LFW_dataset=np.vstack((LFW_masked,LFW_nonmasked))

print("CASIA")
CASIA_masked=load_images_from_folder('./datasets/CASIA-WebFace_masked/webface_masked')
CASIA_nonmasked=load_images_from_folder('./datasets/CASIA-WebFace')

CASIA_masked_labels=np.ones(CASIA_masked.shape[0])
CASIA_nonmasked_labels=np.zeros(CASIA_nonmasked.shape[0])

CASIA_labels=np.hstack((CASIA_masked_labels,CASIA_nonmasked_labels))
CASIA_dataset=np.vstack((CASIA_masked,CASIA_nonmasked))



labels=np.hstack((RFMD_labels,LFW_labels,CASIA_labels))
data=np.vstack((RFMD_dataset,LFW_dataset,CASIA_dataset))

np.savez('data64.npz',data,labels)

