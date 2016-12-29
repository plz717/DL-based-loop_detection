import os
import sys
import zipfile
import glob
from PIL import Image

from six.moves import urllib
import tensorflow as tf
import numpy as np
from . import utils
from .Input import Input


class citycenter(Input):
    """citycenter Faces database input"""

    def __init__(self):
        # Global constants describing the ORL Faces data set.
        self._image_width = 640
        self._image_height = 480
        self._image_depth = 3

        self._num_classes = 0
        self._num_examples_per_epoch_for_train = 12        
        self._num_examples_per_epoch_for_eval = 0
        self._num_examples_per_epoch_for_test = 0

        self._data_dir = "/home/ubuntu/Images_12"

    def num_examples(self, input_type):
        """Returns the number of examples per the specified input_type
        Args:
            input_type: InputType enum
        """
        if not isinstance(input_type, utils.InputType):
            raise ValueError("Invalid input_type, required a valid InputType")

        if input_type == utils.InputType.train:
            return self._num_examples_per_epoch_for_train
        elif input_type == utils.InputType.test:
            return self._num_examples_per_epoch_for_test
        return self._num_examples_per_epoch_for_eval

    def num_classes(self):
        """Returns the number of classes"""
        return self._num_classes

    # adapted from:
    # https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
    def _read(self, filename_queue):
        """Reads and parses examples from MNIST data files.
        Recommendation: if you want N-way read parallelism, call this function
        N times.  This will give you N independent Readers reading different
        files & positions within those files, which will give better mixing of
        examples.
        Args:
            filename_queue: A queue of strings with the filenames to read from.
        Returns:
          An object representing a single example, with the following fields:
              label: an int32 Tensor with the label in the range 0..9.
              image: a [height, width, depth] uint8 Tensor with the image data
        """

        # result = {'image': None, 'label': None}
        result = {'image': None}

        reader = tf.TFRecordReader()
        _, value = reader.read(filename_queue)
        features = tf.parse_single_example(
            value,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string)
                # int64 required
                # 'label': tf.FixedLenFeature([], tf.int64)
            })

        # Convert from a scalar string tensor (whose single string has
        # length IMAGE_WIDHT * self._image_height) to a uint8 tensor with
        # the same shape
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self._image_width * self._image_height * self._image_depth])

        #`Reshape to a valid image
        image = tf.reshape(image, (self._image_height, self._image_width,
                                   self._image_depth))

        # Convert from [0, 255] -> [0, 1] floats.
        image = tf.div(tf.cast(image, tf.float32), 255.0)

        # Convert from [0, 1] -> [-1, 1]
        result["image"] = utils.scale_image(image)

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        # result["label"] = tf.cast(features['label'], tf.int32)

        return result

    def distorted_inputs(self, batch_size):
        """Construct distorted input for ORL Faces training using the Reader ops.
        Args:
            batch_size: Number of images per batch.
        Returns:
            images: Images. 4D tensor of [batch_size, self._image_width, self._image_height, self._image_depth] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """

        with tf.variable_scope("{}_input".format(utils.InputType.train)):
            # Create a queue that produces the filenames to read.
            filename = os.path.join(self._data_dir, 'citycenter.tfrecords')
            filename_queue = tf.train.string_input_producer([filename])
            
            # Read examples from files in the filename queue.
            read_input = self._read(filename_queue)

            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            min_queue_examples = int(self._num_examples_per_epoch_for_train *
                                     min_fraction_of_examples_in_queue)
            print((
                'Filling queue with {} ORL Faces images before starting to train. '
                'This will take a few minutes.').format(min_queue_examples))

            # Generate a batch of images and labels by building up a queue of
            # examples.
            return utils.generate_image_and_label_batch(
                read_input["image"],
                # read_input["label"],
                min_queue_examples,
                batch_size,
                shuffle=True)

    def inputs(self, input_type, batch_size):
        """Construct input for ORL Faces evaluation using the Reader ops.
        Args:
            input_type: InputType enum.
            batch_size: Number of images per batch.
        Returns:
            images: Images. 4D tensor of [batch_size, self._image_width, self._image_height, self._image_depth] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """

        if not isinstance(input_type, utils.InputType):
            raise ValueError("Invalid input_type, required a valid InputType")

        with tf.variable_scope("{}_input".format(input_type)):
            filename = os.path.join(self._data_dir, 'citycenter.tfrecords')
            num_examples_per_epoch = self._num_examples_per_epoch_for_train

            # Create a queue that produces the filenames to read.
            filename_queue = tf.train.string_input_producer([filename])
            print("filename_queue is:",filename_queue)
            # Read examples from files in the filename queue.
            read_input = self._read(filename_queue)

            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            min_queue_examples = int(num_examples_per_epoch *
                                     min_fraction_of_examples_in_queue)

            # Generate a batch of images and labels by building up a queue of
            # examples.
            return utils.generate_image_and_label_batch(
                read_input["image"],
                # read_input["label"],
                min_queue_examples,
                batch_size,
                shuffle=False)

    def maybe_download_and_extract(self):
        """Download and extract the ORL Faces dataset"""

        dest_directory = self._data_dir
        # Convert to Examples and write the result to TFRecords.
        if not tf.gfile.Exists(os.path.join(self._data_dir, 'citycenter.tfrecords')):
            images = []
            # labels = []

            for jpg in glob.glob("{}/*.jpg".format(
                    os.path.join(dest_directory))):
                images.append(np.asarray(Image.open(jpg)))

            # Create dataset object
            dataset = lambda: None
            dataset.num_examples = self._num_examples_per_epoch_for_train
            dataset.images = np.array(images)
            print("dataset.images.shape:",dataset.images.shape)
            # dataset.labels = np.array(labels)
            utils.convert_to_tfrecords(dataset, 'citycenter',self._data_dir)
