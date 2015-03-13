from PIL import Image, ImageOps
import shutil
import os
import random

class_names_file = open("../data/alphabetical_class_list.txt")
class_folders = [line.strip() for line in class_names_file]

OLD_TRIAN_DIR = '../data/train'
NEW_TRAIN_DIR = '../data/train_aug_3'
if os.path.exists(NEW_TRAIN_DIR):
  shutil.rmtree(NEW_TRAIN_DIR)
os.makedirs(NEW_TRAIN_DIR)

NUM_TO_ADD = 10

for i,class_folder in enumerate(class_folders):
  print "Class", i, "/ 121"
  os.makedirs(NEW_TRAIN_DIR + '/' + class_folder)
  images = [item for item in os.listdir(OLD_TRIAN_DIR + '/' + class_folder)]
  for image in images:
    if image == '.DS_Store': continue
    im = Image.open(OLD_TRIAN_DIR + '/' + class_folder + '/' + image)
    im = ImageOps.invert(im)

    for img_num in range(NUM_TO_ADD):
      rotation_degree = img_num*(180/NUM_TO_ADD) + random.randint(0, (180/NUM_TO_ADD))
      new_im = im.rotate(rotation_degree, expand=1)
      new_im = ImageOps.invert(new_im)
      
      new_im.save(NEW_TRAIN_DIR + '/' + class_folder + '/' + image.split('.')[0] + "_" + str(rotation_degree) + '.jpg')
		  
		
