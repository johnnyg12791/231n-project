from PIL import Image
import shutil
import os

class_names_file = open("../data/alphabetical_class_list.txt")
class_folders = [line.strip() for line in class_names_file]

OLD_TRIAN_DIR = '../data/train'
NEW_TRAIN_DIR = '../data/train_aug'
if os.path.exists(NEW_TRAIN_DIR):
  shutil.rmtree(NEW_TRAIN_DIR)
os.makedirs(NEW_TRAIN_DIR)

for i,class_folder in enumerate(class_folders):
  print "Class", i, "/ 121"
  os.makedirs(NEW_TRAIN_DIR + '/' + class_folder)
  images = [item for item in os.listdir(OLD_TRIAN_DIR + '/' + class_folder)]
  for image in images:
    if image == '.DS_Store': continue
    im = Image.open(OLD_TRIAN_DIR + '/' + class_folder + '/' + image)
    for rotation_degrees in [0, 90, 180, 270]:
      new_im = im.rotate(rotation_degrees)
      new_im.save(NEW_TRAIN_DIR + '/' + class_folder + '/' + image.split('.')[0] + "_" + str(rotation_degrees) + '.jpg')
		  
		
		