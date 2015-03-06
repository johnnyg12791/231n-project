import os

directory = '../data/test'

test_file = open("../data/test.txt", "w")

images = [item for item in os.listdir(directory)]
for image in images:
	if image == '.DS_Store': continue
		
	image_file_line = 'test/' + image + "\n"
	test_file.write(image_file_line)

test_file.close()
