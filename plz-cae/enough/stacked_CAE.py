from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf
from my_image_dataset import image_dataset
import pickle


def convert_to_array_queue(files, files_num, height, width, channels):
    # convert files containing multple images in png format to
    # gray_scale images and pack them in
    # image batch[batch_num,height,width]

    filenames = tf.train.match_filenames_once(files)
    filename_queue = tf.train.string_input_producer(filenames)

    # step 3: read, decode and resize images
    reader = tf.WholeFileReader()
    filename, content = reader.read(filename_queue)
    image = tf.image.decode_jpeg(content, channels=channels)
    # for python 3
    resized_image = tf.image.resize_images(image, [height, width])
    # for python 2
    #resized_image = tf.image.resize_images(image, height, width)
    gray_images = tf.image.rgb_to_grayscale(resized_image)

    # step 4: batching
    image_batch = tf.train.batch([gray_images], batch_size=1)
    batch_size = 1
    batch_num = int(files_num / batch_size)

    with tf.Session() as sess_1:
        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess_1, coord=coord)
        image_total = []
        for i in range(batch_num):
            image_tensor = image_batch.eval()  # (1,height,width,1)
            image_array = np.asarray(image_tensor[0])  # (height,width,1)
            image_total.append(image_array)

        # convert list to array
        image_total = np.array(image_total)  # (batch_num,height,W1idth,1))
        image_total = np.multiply(image_total, 1.0 / 255.0)
        # image_total = np.reshape(image_total, (batch_num, height, Width))
        num_examples = image_total.shape[0]

        coord.request_stop()
        coord.join(threads)
    return image_total, num_examples


# train parameters
learning_rate = 0.00001
training_epochs = 10000
train_batch_size = 58
# image patameters
channels = 3  # original depth of input images
DATA_SET_DIR = "/home/plz/slam/Images/"
files_num = 2146  # number of train dataset
height = 480
width = 640
depth = 1  # depth of train_dataset

# convolution parameters
filter_side = 3
stride = 1
filters_number = 32
amount = filter_side - 1
num_layers = 1

files = DATA_SET_DIR + "*.jpg"


# Images:4D tensor of[batch_size, height, width,
# depth size
images, num_examples = convert_to_array_queue(
    files, files_num, height, width, channels)

print("num_examples is:", num_examples)

input_images = tf.placeholder(
    tf.float32, [None, height, width, depth])  # batch,480,640,1
input_x = tf.identity(input_images)

CONV_1_W = [0] * num_layers
CONV_1_b = [0] * num_layers
CONV_2_W = [0] * num_layers
CONV_2_b = [0] * num_layers
DE_W = [0] * num_layers
DE_b = [0] * num_layers
# initialize  the list containing num_layers weight matrix
for i in range(num_layers):
    # encode layer W&b
    # shape1 = [filter_side, filter_side, pad_input.get_shape()[3].value,
             # filters_number]  # 3,3,1,32
    CONV_1_W[i] = tf.Variable(tf.random_normal(
        [3, 3, 1, 32]), dtype=tf.float32)
    CONV_1_b[i] = tf.Variable(tf.zeros([32]), dtype=tf.float32)  # 32
    CONV_2_W[i] = tf.Variable(tf.random_normal(
        [3, 3, 32, 20]), dtype=tf.float32)
    CONV_2_b[i] = tf.Variable(tf.zeros([20]), dtype=tf.float32)  # 20

    # decode layer W&b
    # shape2 = [filter_side, filter_side, filters_number,20]  # 3,3,20,1
    DE_W[i] = tf.Variable(tf.random_normal([3, 3, 20, 1]), dtype=tf.float32)
    DE_b[i] = tf.Variable(tf.zeros([1]), dtype=tf.float32)

for i in range(num_layers):
    # encoder layer
    conv1 = tf.nn.bias_add(tf.nn.conv2d(
        input_x, CONV_1_W[i], [1, 3, 3, 1], 'VALID'), CONV_1_b[i])  # batch,160,213,32
    conv1 = tf.nn.tanh(conv1)
    #print("conv1 is:", conv1)
    # only apply this when training

    pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [
        1, 2, 2, 1], 'VALID')  # batch,80,106,32
    #print("max_pool is:", pool1)

    conv2 = tf.nn.bias_add(tf.nn.conv2d(
        pool1, CONV_2_W[i], [1, 3, 3, 1], 'VALID'), CONV_2_b[i])  # 1,26,35,20

    pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [
        1, 3, 3, 1], 'VALID')  # batch,8,11,20

    encode_dropout = tf.nn.dropout(pool2, 0.5)

    # padding to make it the same size of input_image(batch,480,640,32)
    pad_for_decode = tf.pad(encode_dropout, [[0, 0], [237, 237], [
        316, 315], [0, 0]])  # batch,482,682,20
    #print("pool_pad is :", pad_for_decode)

    # decode layer
    decode_result = tf.nn.bias_add(tf.nn.conv2d(
        pad_for_decode, DE_W[i], [1, stride, stride, 1], 'VALID'), DE_b[i])  # batch,480,640,1
    decode = tf.nn.tanh(decode_result)
    #print("decode is:", decode)
    #print("input_x is:", input_x)

    mse = tf.div(tf.reduce_mean(tf.square(decode - input_x)), 2)
    tf.add_to_collection('losses', mse)
    input_x = tf.stop_gradient(decode)

total_error = tf.add_n(tf.get_collection('losses'))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(total_error)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    images_reshape = np.reshape(images, (files_num, height * width))
    image_batches = image_dataset(images_reshape, images_reshape)
    print("num_examples =", image_batches._num_examples)

    # train cycle
    total_batch = int(files_num / train_batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs = image_batches.next_batch(train_batch_size)
            batch_xs_reshape = np.reshape(
                batch_xs, (train_batch_size, height, width, depth))
            _, c = sess.run([optimizer, total_error], feed_dict={
                            input_images: batch_xs_reshape})

        print("Epoch:", '%04d' % (epoch), "total_error=", "{:.9f}".format(c))

    #print("CONV_1[1] is:", CONV_1_W[1].eval(session=sess))
    #print("CONV_1_b[1] is:", CONV_1_b[1].eval(session=sess))
    coord.request_stop()
    coord.join(threads)
    print("Optimization Finished!")
    saver = tf.train.Saver()
    save_path = saver.save(sess, "stacked_CAE_model.ckpt")
    print("Model saved in file: ", save_path)

    filenames = os.listdir(DATA_SET_DIR)

    features_list = []
    for image in filenames:
        im_test = Image.open(DATA_SET_DIR + image)
        im_test = im_test.convert("L")  # uint8
        im_test = np.reshape(im_test, (-1, height, width, 1))
        im_test = im_test.astype(np.float32)

        input_image = np.multiply(im_test, 1.0 / 255.0)

        input_x = tf.identity(input_image)

        for i in range(num_layers):
            conv1 = tf.nn.bias_add(tf.nn.conv2d(input_x, CONV_1_W[i], [1, 3, 3, 1], 'VALID'), CONV_1_b[i])  # batch,160,213,32
            conv1 = tf.nn.tanh(conv1)
            #print("conv1 is:", conv1)
            # only apply this when training

            pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [
                1, 2, 2, 1], 'VALID')  # batch,80,106,32
            #print("max_pool is:", pool1)

            conv2 = tf.nn.bias_add(tf.nn.conv2d(
                pool1, CONV_2_W[i], [1, 3, 3, 1], 'VALID'), CONV_2_b[i])  # 1,26,35,20
            conv2 = tf.nn.tanh(conv2)
            #print("conv2 is:", conv2)

            pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [
                1, 3, 3, 1], 'VALID')  # batch,8,11,20

            #print("pool_2 is :", pool2)

            # padding to make it the same size of input_image(batch,480,640,32)
            pad_for_decode = tf.pad(pool2, [[0, 0], [237, 237], [
                316, 315], [0, 0]])  # batch,482,642,20
            #print("pool_pad is :", pad_for_decode)

            # decode layer
            decode_result = tf.nn.bias_add(tf.nn.conv2d(
                pad_for_decode, DE_W[i], [1, stride, stride, 1], 'VALID'), DE_b[i])  # batch,480,640,1
            decode = tf.nn.tanh(decode_result)
            #print("decode is:", decode)
            #print("input_x is:", input_x)

            mse = tf.div(tf.reduce_mean(tf.square(decode - input_x)), 2)
            tf.add_to_collection('losses', mse)
            input_x = tf.stop_gradient(decode)

        features_list.append(sess.run(pool2))  # append an array to list
        # print(sess.run(encode))
    # print(features_list)
    #print("feature0 is: ", features_list[0])
    print("feature0 shape is:", (features_list[0].shape))

    # dum features_array_list into a test.txt file
    # with open("test_multi_layers.txt", "wb") as fp:
    #pickle.dump(features_list, fp)
    with open("stacked_CAE_features.txt", "wb") as fp:
        pickle.dump(features_list, fp)
