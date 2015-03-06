import os

directory = '../data/train'

curr_class = 0

labels_file = open("../data/classes.csv", "w")
train_file = open("../data/train.txt", "w")
val_file = open("../data/val.txt", "w")
class_names_file = open("../data/alphabetical_class_list.txt")
class_folders = [line.strip() for line in class_names_file]

for class_folder in class_folders:
	labels_file.write(class_folder + "," + str(curr_class) + "\n")
	images = [item for item in os.listdir(directory + '/' + class_folder)]
	for i, image in enumerate(images):
		if image == '.DS_Store': continue
		
		image_file_line = 'train/' + class_folder + '/' + image + " " + str(curr_class) + "\n"
		if i % 10 == 0:
			val_file.write(image_file_line)
		else:
			train_file.write(image_file_line)		
	curr_class += 1


labels_file.close()
train_file.close()
val_file.close()
