from intensity_normalization.normalize import fcm
from ants import resample_image
from os import listdir
import nibabel as nib

# Directory containing brain and masks
brain_directory = "AD_NL_Registered/"
mask_directory = "MNI152_2009/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"

for scan in listdir(brain_directory):
  # Extract filename
  img_id = scan.split('.')[0]
  print("Checking file:", scan)
  # Load image
  image_loc = brain_directory + scan
  brain = nib.load(image_loc)
  # Load mask
  brain_mask = nib.load(mask_directory)
  # Find white-matter tissue mask
  wm_mask = fcm.find_tissue_mask(brain, brain_mask)
  # Normalise brain MRI
  normalized = fcm.fcm_normalize(brain, wm_mask)
  save_id = img_id + ".nii"
  new_path = "AD_NL_Final/"
  save_path = new_path + save_id
  # Save MRI to new path
  nib.save(normalized, save_path)
  print("File created:", save_id)