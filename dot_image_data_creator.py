import cv2
import numpy as np 
from scipy import spatial
import os
import sys
import time
import random

# for importing mnist:
from tensorflow.contrib.learn.python.learn.datasets import mnist 
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from PIL import Image 

### CHANGE THIS TO SUIT YOUR SYSTEM
dotcounterdir = '/home/shao/Documents/DotCounter/'
datadir = dotcounterdir + 'image_data/'
# Parameters
image_height = 28
image_width = 28
image_channels = 1
dot_radius = 1
dot_colour = (0, 0, 0) # RGB black
num_max_dots_minus_one = 10 # 501
num_images_per_type_train = 6000 # total train set = num_images_per_type_train * num_max_dots_minus_one
num_images_per_type_test = 1000

# for importing mnist:
mnist_image_data_path = dotcounterdir + 'mnist_image_data/'

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

train_dir = "/tmp/data/"
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def create_folder(dirpath):
	if os.path.isdir(dirpath) == False:
		os.makedirs(dirpath)
		os.chmod(dirpath,0o777)


def generate_images(num_images_per_type,subdatadir):
	start_time = time.time()
	create_folder(subdatadir)
	print('Generating images to ' + subdatadir)

	size = (image_height,image_width,image_channels)
	# initialize canvas
	
	# Loop to create all subdirectories of classes/labels of images with a certain number of dots
	for num_dots in range(num_max_dots_minus_one):

		# Loop to create all images of certain number of dots
		for i in range(num_images_per_type):
			canvas = np.full(size, 256, dtype="uint8")
			# make canvas white
			canvas[:] = (255)
			dots_array = np.zeros(shape=(num_dots,2), dtype="uint16")
			counter = 1
			totalcounter = 1
			# Loop to create all dots
			for x in range(num_dots):
				valid_pt = False
				dist = np.full((num_dots,1), 2, dtype="uint16")
								
				while valid_pt == False:
					pt = np.zeros((1,2),dtype="uint16")
					pt[0,0] = np.random.randint(low = (dot_radius + 1), high = (image_width - dot_radius - 1))
					pt[0,1] = np.random.randint(low = (dot_radius + 1), high = (image_height - dot_radius - 1))
					dist = spatial.distance.cdist(dots_array,pt)
					totalcounter = totalcounter + 1
					if (all(d>3 for d in dist)):
						valid_pt = True
						dots_array[x] = pt
												
						#print('valid pt = ')
						#print(pt)
						#print('dots array = ')
						#dot_colour = (np.random.randint(low = 0, high = 255),np.random.randint(low = 0, high = 255),np.random.randint(low = 0, high = 255))
						cv2.circle(canvas, tuple((pt[0,0],pt[0,1])), dot_radius, list(dot_colour), -1)
			
			#print('dots array = ')
			#print(dots_array)
			
			# check if subdirpath is created, if not create it
			subdirpath = subdatadir + str(num_dots)
			#print(subdirpath)
			if os.path.isdir(subdirpath) == False:
				os.makedirs(subdirpath)
				os.chmod(subdirpath,0o777)

			filepath = (subdirpath + '/' + str(num_dots) + '_' + str(i) + '.jpg')
			cv2.imwrite(filepath, canvas)
			print('Image ' + str(i) + ' of class -' + str(num_dots) + ' dot images- created')	

	print('Time taken to generate images to ' + subdatadir)		
	print("--- %s seconds ---" % (time.time() - start_time))


def create_images_labels_textfile(Labels_File_Path,Input_Data_Directory):
    with open(Labels_File_Path, "a") as a:
        for path, subdirs, files in os.walk(Input_Data_Directory):      
            for filename in files:
                f = os.path.join(path, filename)
                if f.endswith(".jpg"):
                    label = path.split("/")
                    a.write(str(f) + " " + label[(len(label) - 1)] + os.linesep) 
    print(Labels_File_Path + ' created!')



def randomize_textfile(Labels_File_Path):
    with open(Labels_File_Path,'r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(Labels_File_Path,'w') as target:
        for _, line in data:
            target.write( line )
    print(Labels_File_Path + ' randomized!')

def generate_mnist_jpg(subdatadir, source_image_path, source_label_path):
	create_folder(subdatadir)
	local_file = base.maybe_download(source_image_path, train_dir,
									SOURCE_URL + source_image_path)
	with open(local_file, 'rb') as f:
		images = mnist.extract_images(f)
	

	local_file = base.maybe_download(source_label_path, train_dir,
                                   SOURCE_URL + source_label_path)
	with open(local_file, 'rb') as f:
		labels = mnist.extract_labels(f, one_hot=False)

	for img in range(labels.size):
		subdirpath = subdatadir + str(labels[img])
		create_folder(subdirpath)

		filepath = (subdirpath + '/' + str(labels[img]) + '_' + str(img) + '.jpg')
		im = Image.fromarray(images[img,:,:,0])
		im.save(filepath)




# MAIN CODE

savednetsdir = dotcounterdir + 'saved_nets'
create_folder(savednetsdir)

# dot
create_folder(datadir)
subdatadir = datadir + 'TEST_IMAGES/'
generate_images(num_images_per_type_test, subdatadir)

file_path = subdatadir + 'TEST_IMAGES_LABELS_FILE.txt'
create_images_labels_textfile(file_path,subdatadir)
randomize_textfile(file_path)

subdatadir = datadir + 'TRAIN_IMAGES/'
generate_images(num_images_per_type_train, subdatadir)

file_path = subdatadir + 'TRAIN_IMAGES_LABELS_FILE.txt'
create_images_labels_textfile(file_path,subdatadir)
randomize_textfile(file_path)

# mnist
create_folder(mnist_image_data_path)

subdatadir = mnist_image_data_path + 'TEST_IMAGES/'
generate_mnist_jpg(subdatadir, TEST_IMAGES, TEST_LABELS)

file_path = subdatadir + 'TEST_IMAGES_LABELS_FILE.txt'
create_images_labels_textfile(file_path,subdatadir)
randomize_textfile(file_path)

subdatadir = mnist_image_data_path + 'TRAIN_IMAGES/'
generate_mnist_jpg(subdatadir, TRAIN_IMAGES, TRAIN_LABELS)

file_path = subdatadir + 'TRAIN_IMAGES_LABELS_FILE.txt'
create_images_labels_textfile(file_path,subdatadir)
randomize_textfile(file_path)

sys.exit()

