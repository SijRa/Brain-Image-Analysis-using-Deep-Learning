from ants import registration, image_read, image_write
from os import listdir

mri_directory = "AD_NL_Brains/"
mask_directory = "AD_NL_Masks/"

mri_save_dir = "AD_NL_Registered/"

template_loc = "MNI152_2009/mni_152_brain.nii"
template = image_read(template_loc)

for scan in listdir(mri_directory):
  id = scan.split('+')[0]
  img_path = mri_directory + scan
  mask_path = mask_directory + id + ".nii"
  image = image_read(img_path, reorient=True)
  mask = image_read(mask_path, reorient=True)
  registered_dict = registration(fixed=template, moving=image, type_of_transform="SyNAggro")
  filename = mri_save_dir + id + ".nii"
  image_write(registered_dict['warpedmovout'], filename=filename)
  print("Registered:",scan)