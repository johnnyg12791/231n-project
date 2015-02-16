from PIL import Image
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

directory = '../data/train'



X_train = np.zeros((30336, 64, 64))
Y_train = np.zeros((30336, 1))

i = 0
curr_class = 0

#0,2,18, 25

class_folders = [item for item in os.listdir(directory) if not os.path.isfile(os.path.join(directory, item))]
for class_folder in class_folders:
	images = [item for item in os.listdir(directory + '/' + class_folder)]
	for image in images:
		if image == '.DS_Store': continue
		im = Image.open(directory + '/' + class_folder + '/' + image)

		im = im.resize((64,64)) # TODO: explore how effective this is. SOME IMAGES ARE CROPPED, this is bad

		curr = np.asarray(im.convert('L'))
		X_train[i, :, :] = curr
		Y_train[i] = curr_class

		i += 1
	curr_class += 1



mean_image = np.mean(X_train, axis=0)
# DISPLAYING THE MEAN IMAGE
#plt.imshow(mean_image, cmap = cm.Greys_r)
#plt.show()

'''
# DISPLAYING 2 MEAN IMAGES FOR CLASSES 2 AND 18

indexes1 = [x for x in range(len(Y_train)) if Y_train[x] == 2 ]
indexes2 = [x for x in range(len(Y_train)) if Y_train[x] == 18 ]

mean_image = np.mean(X_train[indexes1, :, :], axis=0)
plt.imshow(mean_image, cmap = cm.Greys_r)
plt.show()

mean_image = np.mean(X_train[indexes2, :, :], axis=0)
plt.imshow(mean_image, cmap = cm.Greys_r)
plt.show()
'''



# second: subtract the mean image from train and test data
plt.imshow(X_train[0, :, :], cmap = cm.Greys_r)
plt.show()
X_train -= mean_image


plt.imshow(X_train[0, :, :], cmap = cm.Greys_r)
plt.show()
X_train /= np.std(X_train, axis=0)
plt.imshow(X_train[0, :, :], cmap = cm.Greys_r)
plt.show()

std = np.std(X_train, axis=0)
plt.imshow(std, cmap = cm.Greys_r)
plt.show()

print X_train.shape
print Y_train.shape

print "Done."