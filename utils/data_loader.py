#!/usr/bin/env python

import random
from os import listdir
import numpy as np
import pandas as pd
import nibabel as nib
from SimpleITK import ReadImage
from dltk.io.preprocessing import *

class MRI_Loader:
  
  def __init__(self, target_shape, load_size=None):
    self.mri_path = '../ADNI_volumes_custom/'
    self.csv_path = '../ADNI_clinical_data/LP_ADNIMERGE_CLEANED.csv'
    self.target_shape = target_shape
    self.load_size = None if load_size == None else load_size
  
  def Get_Class_Information(self, _list, skipped_files):
    print("\n--- CLASS BALANCE ---")
    num_files = len(_list)
    AD, MCI, CN = 0,0,0
    pMCI, sMCI = 0,0
    highRisk, lowRisk = 0,0
    for _file in _list:
      filename_arr = _file.split('_')
      if len(filename_arr) == 7:
        _class = filename_arr[5]
      else:
        _class = filename_arr[4]
      if _class == "AD":
        AD += 1
      elif _class == "MCI":
        MCI += 1
      elif _class == "CN":
        CN += 1
      else:
        print(_class)
        print("DATA ERROR: Incorrect class")
        exit()
      _pVsClass = filename_arr[3]
      if _pVsClass == 'unstableMCI':
        pMCI += 1
        if filename_arr[4] == 'HR':
          highRisk += 1
        elif filename_arr[4] == 'LR':
          lowRisk += 1
      elif _pVsClass == 'stableMCI':
        sMCI += 1
      else:
        print(_pVsClass)
        print("DATA ERROR: Incorrect Format")
        exit()
    print("AD: " + str(AD) + "\n" + str(AD*100/num_files) + "%")
    print("MCI: " + str(MCI) + "\n" + str(MCI*100/num_files) + "%")
    print("CN: " + str(CN) + "\n" + str(CN*100/num_files) + "%\n")
    print("pMCI: " + str(pMCI) + "\n" + str(pMCI*100/num_files) + "%")
    print("High Risk (pMCI): " + str(highRisk) + "\n" + str(highRisk*100/pMCI)+ "%")
    print("Low Risk (pMCI): " + str(lowRisk) + "\n" + str(lowRisk*100/pMCI)+ "%\n")
    print("sMCI: " + str(sMCI) + "\n" + str(sMCI*100/num_files) + "%\n")
    print("Skipped Files:",skipped_files)
  
  def Shuffle_List(self, _list):
    if any(_list):
      random.shuffle(_list)
    
  def Get_Filenames(self):
    file_names = sorted(listdir(self.mri_path))
    self.Shuffle_List(file_names)
    return file_names
  
  def Generate_Label(self, _class, _pVsClass, _riskClass):
    #class_dict = {"AD":np.array([0, 0, 1]), "MCI":np.array([0, 1, 0]), "CN":np.array([1, 0, 0])}
    class_dict = {"AD":2, "MCI":1, "CN":0}
    pVs_dict = {"unstableMCI":1, "stableMCI":0}
    risk_dict = {"HR":2, "LR":1, 0:0}
    return class_dict[_class], pVs_dict[_pVsClass], risk_dict[_riskClass]
  
  def Shape_Constraint(self, _file):
    image_data = ReadImage(self.mri_path + _file)
    return True if image_data.GetSize() == self.target_shape else False
  
  def Extract_Filename(self, _filearray):
    _idarray = _filearray[:3]
    _id = "_"
    _id = _id.join(_idarray)
    _pVsClass = _filearray[3]
    _riskClass = 0 
    _class = None
    _date = None
    if len(_filearray) == 7:
        ## pMCI
        _riskClass = _filearray[4]
        _class = _filearray[5]
        _date = _filearray[6].split('.')[0]
    elif len(_filearray) == 6:
        ##sMCI
        _riskClass = 0
        _class = _filearray[4]
        _date = _filearray[5].split('.')[0]
    return _id, _date, _class, _pVsClass, _riskClass
  
  def Extract_Clinical(self, _id, _date, clinical_data):
    feature = np.array([])
    patient_data = clinical_data.drop(columns=["PTID","EXAMDATE"])  
    feature = np.append(feature, patient_data[(clinical_data.PTID == _id) & (clinical_data.EXAMDATE == _date)].values.flatten(), axis=0)
    return feature
    
  def Extract_Data(self, filenames):
    mri_features, clinical_features = [],[]
    class_labels, pVs_labels, risk_labels = [],[],[]
    loaded_files = []
    skipped_files = 0
    print("\n--- COLLECTING MRI ---")
    filenames = filenames[:self.load_size] if self.load_size != None else filenames
    clinical_data = pd.read_csv(self.csv_path)
    for _file in filenames:
      if self.Shape_Constraint(_file):
        _filearray = _file.split('_')
        _id, _date, _class, _pVsClass, _riskClass = self.Extract_Filename(_filearray)
        # Extract clinical features
        clinical_feature = self.Extract_Clinical(_id, _date, clinical_data.copy())
        if (clinical_feature.size == 0):
          print("DATA WARNING: Clinical data not found, skipping file")
          skipped_files += 1
          continue
        print("LOADING MRI - " + _file)
        # Load MRI
        image_data = nib.load(self.mri_path + _file).get_data().astype('float32')
        normalised_data = whitening(image_data) # z-score normalisation
        normalised_mri = np.expand_dims(normalised_data, axis=-1)
        # Generate Labels
        classLabel, pVsLabel, riskLabel = self.Generate_Label(_class, _pVsClass, _riskClass)
        class_labels.append(classLabel)
        pVs_labels.append(pVsLabel)
        risk_labels.append(riskLabel)
        clinical_features.append(clinical_feature)
        mri_features.append(normalised_mri)
        loaded_files.append(_file)
    self.Get_Class_Information(loaded_files, skipped_files)
    return np.asarray(mri_features), np.asarray(clinical_features), np.asarray(class_labels), np.asarray(pVs_labels), np.asarray(risk_labels)
      
  def Load_Data(self):
    features = {'mri':[], 'clinical':[]}
    labels = {'class':[], 'pvs':[], 'risk':[]}
    filenames = self.Get_Filenames()
    mri_features, clinical_features, class_labels, pVs_labels, risk_labels = self.Extract_Data(filenames)
    features['mri'] = mri_features
    features['clinical'] = clinical_features
    labels['class'] = class_labels
    labels['pvs'] = pVs_labels
    labels['risk'] = risk_labels
    return features, labels