from ants import registration, image_read, image_write, resample_image, crop_image
from os import listdir

mri_directory = "ADNI_baseline_raw/"

template_loc = "MNI152_2009/mni_icbm152_t1_tal_nlin_sym_09a.nii"
template = image_read(template_loc)
template = resample_image(template, (192, 192, 160), True, 4)
#template = crop_image(template)

for scan in listdir(mri_directory):
  id = scan.split('.')[0]
  filename = "ADNI_original_registered/" + id + ".nii"
  img_path = mri_directory + scan
  image = image_read(img_path, reorient=True)
  if image.shape[1] != 192:
    print("- Resampling -")
    image = resample_image(image, (192, 192, 160), True, 4)
  registered_dict = registration(fixed=template, moving=image, type_of_transform="SyNRA")
  #img = crop_image(registered_dict['warpedmovout'])
  image_write(registered_dict['warpedmovout'], filename=filename)
  print("Registered:",scan)