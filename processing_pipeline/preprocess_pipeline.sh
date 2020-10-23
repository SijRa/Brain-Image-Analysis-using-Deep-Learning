#!/usr/bin/bash

########## N4 Bisa Field Correction; Brain + Mask Extraction; Registration; Intensity Normalisation ##########

###############################################################################
# Load cmake and ready path for N4BiasFieldCorrection
module load cmake/3.13.0
export PATH=/home/554282/Packages/antsInstallExample/install/bin:$PATH
export ANTSPATH=/home/554282/Packages/antsInstallExample/install/bin

## N4 Bias Field Correction
bash RunN4Bias.sh
###############################################################################
## Brain Extraction

# Scan location
MRI_SCANS=/home/554282/Deep-Learning-Brain-Image-Analysis/AD_NL_N4/*

# Extract brain tissue and generate brain mask from MRI
for FILE in $MRI_SCANS
do
  echo "Processing $FILE"
  deepbrain-extractor -i $FILE -o AD_NL_Brains/$(basename $FILE)
done
echo "Processing Complete"
###############################################################################
# Rename extracted brains + masks
python rename.py
###############################################################################
## Registration
python register.py
###############################################################################
## Intensity Normalisation - Fuzzy C-means-based
python normalise.py 
###############################################################################