#from sklearn.utils import shuffle

from pcanet import PCANet
import tensorflow as tf
import numpy as np
import random
np.set_printoptions(threshold='nan')
# convert files containing multple images in png format to
# gray_scale images and pack them in
# image batch[batch_num,height,width]


def convert_to_array_queue(files, files_num, height, width, channels):
    filenames = tf.train.match_filenames_once(files)
    filename_queue = tf.train.string_input_producer(filenames)

    # step 3: read, decode and resize images
    reader = tf.WholeFileReader()
    filename, content = reader.read(filename_queue)
    image = tf.image.decode_png(content, channels=channels)
    resized_image = tf.image.resize_images(image, [height, width])
    gray_images = tf.image.rgb_to_grayscale(resized_image)

    # step 4: Batching
    image_batch = tf.train.batch([gray_images], batch_size=1)
    batch_size = 1
    batch_num = int(files_num / batch_size)

    with tf.Session() as sess_1:
        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess_1, coord=coord)
        image_total = []
        for i in range(batch_num):
            image_tensor = image_batch.eval()  # (1,heightheight,width,1)
            image_array = np.asarray(image_tensor[0])  # (height,widthwidth,1)
            image_total.append(image_array)
        # convert list to array
        image_total = np.array(image_total)  # (batch_num,height,width,1))
        image_total = np.reshape(image_total, (batch_num, height, width))
        num_examples = image_total.shape[0]

        coord.request_stop()
        coord.join(threads)
    return image_total, num_examples


height = 60
width = 80
channels = 3
files = "/home/ubuntu/PCANet-python3/rgb_png/*.png"
files_num = 96

images, num_examples = convert_to_array_queue(
    files=files, files_num=files_num, height=height,
    width=width, channels=channels)
images = random.shuffle(images)
print(images)
print(num_examples)
print(images.shape)

pcanet = PCANet(
    image_shape=(60, 80),
    filter_shape_l1=2, step_shape_l1=1, n_l1_output=4,
    filter_shape_l2=2, step_shape_l2=1, n_l2_output=4,
    block_shape=2
)

print("has excuted PCANET function")

pcanet.validate_structure()

print("has excuted pcanet.validate_structure")

pcanet.fit(images)

print("has excuted pacnet.fit")

X_train = pcanet.transform(images)

print("has excuted pcanet.transform")

print(X_train)
print(X_train.shape)
