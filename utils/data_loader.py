#!/usr/bin/env python

import random
from os import listdir
import numpy as np
import nibabel as nib
from SimpleITK import ReadImage
from dltk.io.preprocessing import *

class MRI_Loader:
  
  def __init__(self, target_shape, load_size=None):
    self.mri_path = '../ADNI_volumes_customtemplate_float32/'
    self.xls_path = '../ADNI_clinical_data/'
    self.target_shape = target_shape
    self.load_size = None if load_size == None else load_size
  
  def Get_Class_Information(self, _list):
    print("\n--- CLASS INFORMATION ---")
    num_files = len(_list)
    AD, NL, MCI = 0,0,0
    for _file in _list:
      filename_arr = _file.split('_')
      if filename_arr[0] == 'stableAD':
        AD += 1
      elif filename_arr[0] == 'stableNL':
        NL += 1
      elif filename_arr[0] == 'stableMCI':
        MCI += 1
    print("AD: " + str(AD) + "\n" + str(AD*100/num_files) + "%\n")
    print("Control: " + str(NL) + "\n" + str(NL*100/num_files)+ "%\n")
    print("MCI: " + str(MCI) + "\n" + str(MCI*100/num_files)+ "%\n")
  
  def Shuffle_List(self, _list):
    if any(_list):
      random.shuffle(_list)
    
  def Get_Filenames(self):
    file_names = sorted(listdir(self.mri_path))
    self.Get_Class_Information(file_names)
    self.Shuffle_List(file_names)
    return file_names
  
  def Generate_Label(self, p_class):
    label_dict = {"stableAD":np.array([0,0,1]),
              "stableMCI":np.array([0,1,0]),
              "stableNL":np.array([1,0,0])}
    return label_dict[p_class]
  
  def Shape_Constraint(self, _file):
    image_data = ReadImage(self.mri_path + _file)
    return True if image_data.GetSize() == self.target_shape else False
    
  def Extract_MRI(self, filenames):
    features, labels = [], []
    print("\n--- COLLECTING MRI ---")
    filenames = filenames[:self.load_size] if self.load_size != None else filenames
    for _file in filenames:
      print("LOADING MRI - " + _file)
      image_data = nib.load(self.mri_path + _file).get_data().astype('float32')
      if self.Shape_Constraint(_file):
        normalised_data = whitening(image_data) # z-score normalisation
        normalised_data = np.expand_dims(normalised_data, axis=-1)
        filename_arr = _file.split('_')
        label = self.Generate_Label(filename_arr[0])
        features.append(normalised_data)
        labels.append(label)
    return features, labels
      
  def Load_Data(self):
    data = {'features':[], 'labels':[]}
    filenames = self.Get_Filenames()
    features, labels = self.Extract_MRI(filenames)
    data['features'] = features
    data['labels'] = labels
    return data