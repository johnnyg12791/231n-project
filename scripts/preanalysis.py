import os
import matplotlib.pyplot as plt

directory = '../data/train'

class_folders = [item for item in os.listdir(directory) if not os.path.isfile(os.path.join(directory, item))]
counts = []
for class_folder in class_folders:
	count = len([item for item in os.listdir(directory + '/' + class_folder)])
	if count>1000:
		print class_folder, ':', count

	counts.append(count)


plt.hist(counts, bins=25)
plt.show()	

#trichodesmium_puff : 1979
#chaetognath_other : 1934
#copepod_cyclopoid_oithona_eggs : 1189
#protist_other : 1172

#unknown_blobs_and_smudges : 317
#unknown_sticks : 175
#unknown_unclassified : 425