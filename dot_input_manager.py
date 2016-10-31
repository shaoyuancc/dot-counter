
# ==============================================================================

"""Functions for downloading and reading dots data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from scipy import misc


def read_labeled_image_list(image_list_file, one_hot=False,num_classes=10):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: label will be pasted after each line
    Returns:
       A numpy image_array of shape [number of images,28,28,number of channels]
       A numpy labels_array of shape [number of images]
    """

    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    
    for line in f:
        filename, label = line[:-1].split(' ')
        labels.append(int(label))
        filenames.append(str(filename))
    labels_array = numpy.asarray(labels, dtype=numpy.uint8)
    
    if one_hot:
      labels_array = dense_to_one_hot(labels_array, num_classes)
    
    image_array = numpy.zeros([len(filenames),28,28],dtype=numpy.uint8)
    for f in range(len(filenames)):

        image_array[f] = misc.imread(filenames[f])
    
    # Add new Axis to image array so it will have shape [number of images,28,28,number of channels]
    image_array = image_array[:,:,:,numpy.newaxis]
    
    return image_array, labels_array


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


class DataSet(object):

  def __init__(self,
               images,
               labels,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    
    
    assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir ='/home/shao/Documents/DotCounter/image_data/',
                   one_hot=False, num_classes = 10,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000):
  
  TRAIN_IMAGES_LABELS_FILE = train_dir + 'TRAIN_IMAGES/TRAIN_IMAGES_LABELS_FILE.txt'
  TEST_IMAGES_LABELS_FILE = train_dir + 'TEST_IMAGES/TEST_IMAGES_LABELS_FILE.txt'
  
  train_images, train_labels = read_labeled_image_list(TRAIN_IMAGES_LABELS_FILE,one_hot,num_classes)
  test_images, test_labels = read_labeled_image_list(TEST_IMAGES_LABELS_FILE,one_hot,num_classes)
  

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_images,
                       validation_labels,
                       dtype=dtype,
                       reshape=reshape)
  test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=test)

#def num_train_images(train_dir ='/home/shao/Documents/dots28by28/image_data/'):
 #   TRAIN_IMAGES_LABELS_FILE = train_dir + 'TRAIN_IMAGES/TRAIN_IMAGES_LABELS_FILE.txt'
  #  with open(TRAIN_IMAGES_LABELS_FILE) as f:
   # 	return(sum(1 for _ in f))


