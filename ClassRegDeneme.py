import tensorflow as tf
import numpy as np
import datasatReg
import math


#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.probs = self.fcn11  #self.conv8_1 #self.fc3l
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')  #[3, 3, 3, 64] 3x3 filter size 3 channel 64 filter
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')  # 1 x 1 stride ilk ve son 1 herzaman 1
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')  # 64 tane filter var o yuzden 64 bias lazim
            out = tf.nn.bias_add(conv, biases)  # conv += bias in kisaltilmis hali
            # self.conv1_1 = tf.nn.relu(out, name=scope)  # normalde bu ama onlarin odev isterleri tanh sanirim sonra degistiririz
            self.conv1_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')  #[3, 3, 64, 64] 3x3 filter size 64 channel 64 filter
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')   # 1 x 1 stride ilk ve son 1 herzaman 1
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),  trainable=True, name='biases')  # 64 tane filter var o yuzden 64 bias lazim
            out = tf.nn.bias_add(conv, biases)  # conv += bias in kisaltilmis hali
            # self.conv1_2 = tf.nn.relu(out, name=scope)
            self.conv1_2 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1],  strides=[1, 2, 2, 1],  padding='SAME', name='pool1')  # 2x2 filter 2 stride( 2x2 demek x and y)

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,  stddev=1e-1), name='weights')  #[3, 3, 64, 128] 3x3 filter size inputta 64 channelimiz var 128 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')   # 1 x 1 stride ilk ve son 1 herzaman 1
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),  trainable=True, name='biases')  # 128 tane filter var o yuzden 128 bias lazim
            out = tf.nn.bias_add(conv, biases)  # conv += bias in kisaltilmis hali
            # self.conv2_1 = tf.nn.relu(out, name=scope)
            self.conv2_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,  stddev=1e-1), name='weights')  #[3, 3, 128, 128] 3x3 filter size inputta 128 channelimiz var 128 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')  # 1 x 1 stride ilk ve son 1 herzaman 1
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')  # 128 tane filter var o yuzden 128 bias lazim
            out = tf.nn.bias_add(conv, biases)   # conv += bias in kisaltilmis hali
            # self.conv2_2 = tf.nn.relu(out, name=scope)
            self.conv2_2 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')  # 2x2 filter 2 stride( 2x2 demek x and y)

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,  stddev=1e-1), name='weights')  #[3, 3, 128, 256] 3x3 filter size inputta 128 channelimiz var 256 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')  # 1 x 1 stride ilk ve son 1 herzaman 1
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),  trainable=True, name='biases')  # 256 tane filter var o yuzden 128 bias lazim
            out = tf.nn.bias_add(conv, biases)  # conv += bias in kisaltilmis hali
            # self.conv3_1 = tf.nn.relu(out, name=scope)
            self.conv3_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')  #[3, 3, 256, 256] 3x3 filter size inputta 256 channelimiz var 256 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),  trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv3_2 = tf.nn.relu(out, name=scope)
            self.conv3_2 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,  stddev=1e-1), name='weights')  #[3, 3, 256, 256] 3x3 filter size inputta 128 channelimiz var 256 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv3_3 = tf.nn.relu(out, name=scope)
            self.conv3_3 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,  ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  padding='SAME',  name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1), name='weights')  #[3, 3, 256, 512] 3x3 filter size inputta 256 channelimiz var 512 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),  trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv4_1 = tf.nn.relu(out, name=scope)
            self.conv4_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')  #[3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv4_2 = tf.nn.relu(out, name=scope)
            self.conv4_2 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,  stddev=1e-1), name='weights')  #[3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagi
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),  trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv4_3 = tf.nn.relu(out, name=scope)
            self.conv4_3 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,  ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',  name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights') #[3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagi
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv5_1 = tf.nn.relu(out, name=scope)
            self.conv5_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 512], dtype=tf.float32,  stddev=1e-1), name='weights') #[3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagi
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv5_2 = tf.nn.relu(out, name=scope)
            self.conv5_2 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights') #[3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagi
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),  trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv5_3 = tf.nn.relu(out, name=scope)
            self.conv5_3 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,  ksize=[1, 2, 2, 1],  strides=[1, 2, 2, 1],  padding='SAME',  name='pool4')


        # conv6_1
        with tf.name_scope('conv6_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([7, 7, 512, 4096], dtype=tf.float32, stddev=1e-1), name='weights') #[3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagi
            conv = tf.nn.conv2d(self.pool5, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),  trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv5_3 = tf.nn.relu(out, name=scope)
            self.conv6_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv7_1
        with tf.name_scope('conv7_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 4096], dtype=tf.float32, stddev=1e-1), name='weights')  # [3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagi
            conv = tf.nn.conv2d(self.conv6_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv5_3 = tf.nn.relu(out, name=scope)
            self.conv7_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv8_1
        with tf.name_scope('conv8_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([1, 1, 4096, 28], dtype=tf.float32, stddev=1e-1), name='weights')  # [3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagi
            conv = tf.nn.conv2d(self.conv7_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[28], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv5_3 = tf.nn.relu(out, name=scope)
            self.conv8_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
        fcn9 = tf.layers.conv2d_transpose(self.conv8_1, filters=self.pool4.get_shape().as_list()[-1], kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")
        # Add a skip connection between current final layer fcn8 and 4th layer

        fcn9_skip_connected = tf.add(fcn9, self.pool4, name="fcn9_plus_vgg_layer4")

        # Upsample again
        fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=self.pool3.get_shape().as_list()[-1], kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

        # Add skip connection
        fcn10_skip_connected = tf.add(fcn10, self.pool3, name="fcn10_plus_vgg_layer3")

        # Upsample again
        self.fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=14, kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

        tempdenme = 5





class vgg16Reg:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        # self.probs = tf.nn.softmax(self.fc3l)
        # print(self.pool4.get_shape())
        self.probs = self.fc3l
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def convlayers(self):
        self.parameters = []

        images = self.imgs

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 17, 64], dtype=tf.float32, stddev=1e-1), name='weights')  #[3, 3, 3, 64] 3x3 filter size 3 channel 64 filter
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')  # 1 x 1 stride ilk ve son 1 herzaman 1
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')  # 64 tane filter var o yuzden 64 bias lazim
            out = tf.nn.bias_add(conv, biases)  # conv += bias in kisaltilmis hali
            # self.conv1_1 = tf.nn.relu(out, name=scope)  # normalde bu ama onlarin odev isterleri tanh sanirim sonra degistiririz
            self.conv1_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')  #[3, 3, 64, 64] 3x3 filter size 64 channel 64 filter
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')   # 1 x 1 stride ilk ve son 1 herzaman 1
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),  trainable=True, name='biases')  # 64 tane filter var o yuzden 64 bias lazim
            out = tf.nn.bias_add(conv, biases)  # conv += bias in kisaltilmis hali
            # self.conv1_2 = tf.nn.relu(out, name=scope)
            self.conv1_2 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1],  strides=[1, 2, 2, 1],  padding='SAME', name='pool1')  # 2x2 filter 2 stride( 2x2 demek x and y)

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,  stddev=1e-1), name='weights')  #[3, 3, 64, 128] 3x3 filter size inputta 64 channelimiz var 128 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')   # 1 x 1 stride ilk ve son 1 herzaman 1
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),  trainable=True, name='biases')  # 128 tane filter var o yuzden 128 bias lazim
            out = tf.nn.bias_add(conv, biases)  # conv += bias in kisaltilmis hali
            # self.conv2_1 = tf.nn.relu(out, name=scope)
            self.conv2_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,  stddev=1e-1), name='weights')  #[3, 3, 128, 128] 3x3 filter size inputta 128 channelimiz var 128 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')  # 1 x 1 stride ilk ve son 1 herzaman 1
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')  # 128 tane filter var o yuzden 128 bias lazim
            out = tf.nn.bias_add(conv, biases)   # conv += bias in kisaltilmis hali
            # self.conv2_2 = tf.nn.relu(out, name=scope)
            self.conv2_2 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')  # 2x2 filter 2 stride( 2x2 demek x and y)

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,  stddev=1e-1), name='weights')  #[3, 3, 128, 256] 3x3 filter size inputta 128 channelimiz var 256 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')  # 1 x 1 stride ilk ve son 1 herzaman 1
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),  trainable=True, name='biases')  # 256 tane filter var o yuzden 128 bias lazim
            out = tf.nn.bias_add(conv, biases)  # conv += bias in kisaltilmis hali
            # self.conv3_1 = tf.nn.relu(out, name=scope)
            self.conv3_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')  #[3, 3, 256, 256] 3x3 filter size inputta 256 channelimiz var 256 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),  trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv3_2 = tf.nn.relu(out, name=scope)
            self.conv3_2 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,  stddev=1e-1), name='weights')  #[3, 3, 256, 256] 3x3 filter size inputta 128 channelimiz var 256 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv3_3 = tf.nn.relu(out, name=scope)
            self.conv3_3 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,  ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  padding='SAME',  name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1), name='weights')  #[3, 3, 256, 512] 3x3 filter size inputta 256 channelimiz var 512 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),  trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv4_1 = tf.nn.relu(out, name=scope)
            self.conv4_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')  #[3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagiz
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv4_2 = tf.nn.relu(out, name=scope)
            self.conv4_2 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,  stddev=1e-1), name='weights')  #[3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagi
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),  trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv4_3 = tf.nn.relu(out, name=scope)
            self.conv4_3 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,  ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',  name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights') #[3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagi
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv5_1 = tf.nn.relu(out, name=scope)
            self.conv5_1 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,  stddev=1e-1), name='weights') #[3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagi
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv5_2 = tf.nn.relu(out, name=scope)
            self.conv5_2 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights') #[3, 3, 512, 512] 3x3 filter size inputta 512 channelimiz var 512 filter uyguluyacagi
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),  trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            # self.conv5_3 = tf.nn.relu(out, name=scope)
            self.conv5_3 = tf.nn.tanh(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,  ksize=[1, 2, 2, 1],  strides=[1, 2, 2, 1],  padding='SAME',  name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))

            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],  dtype=tf.float32,  stddev=1e-1), name='weights')  #pool5 in output sayisi 4096 hidden node sayisi
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),  trainable=True, name='biases')  # hidden node sayisi kadar bias olacak
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)  # pool5 den gelen inputlarla weightleri carpip biaslari topluyor
            # self.fc1 = tf.nn.relu(fc1l)
            self.fc1 = tf.nn.tanh(fc1l)
            self.parameters += [fc1w, fc1b]

            self.probs = self.fc1

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],  dtype=tf.float32,  stddev=1e-1), name='weights')  # 1. hidden 4096 nodeu ile 2. hidden nodes lar ile esliyor
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),  trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            # self.fc2 = tf.nn.relu(fc2l)
            self.fc2 = tf.nn.tanh(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 28],  dtype=tf.float32,  stddev=1e-1), name='weights')  # 2.hidden to output layer
            fc3b = tf.Variable(tf.constant(1.0, shape=[28], dtype=tf.float32),  trainable=True, name='biases')
            fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.fc3l = tf.nn.softmax(fc3l)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i <= 27:
                print(i, k, np.shape(weights[k]))
                sess.run(self.parameters[i].assign(weights[k]))

def createheatmaps2(label):

    ph_num_rows = label.shape[0]
    labels_one_hots = np.zeros((ph_num_rows, 224, 224, 14))

    for l in range(0, ph_num_rows):
        for k in range(0, 14):
            for i in range(0, 224):
                for j in range(0, 224):
                    dist = math.hypot(i - label[l][k], j - label[l][k+14])
                    if abs(dist) > 10:
                        labels_one_hots[l][i][j][k] = 0
                    else:
                        labels_one_hots[l][i][j][k] = 1
    return labels_one_hots

def createheatmaps(x_shaped, y_): # x_shaped: input, y_ output heatmpsler : y truthlar

    xheatmaps0 = tf.reshape(y_, [-1, 14, 224, 224])
    xheatmaps1 = tf.reshape(x_shaped, [-1, 3, 224, 224])
    xheatmaps2 = tf.concat([xheatmaps0, xheatmaps1], 1)
    xheatmaps = tf.reshape(xheatmaps2, [-1, 224, 224, 17])

    return xheatmaps


def run_cnn():
    mnist = datasatReg.read_data_sets("/regresiion/", one_hot=True)

    with tf.Session() as sess:
        # Python optimisation variables
        learning_rate = 0.0001
        epochs = 10
        batch_size = 50

        # bizim input 224x224x3 = 150528
        x = tf.placeholder(tf.float32, [None, 150528])

        # image size 224x224x3
        x_shaped = tf.reshape(x, [-1, 224, 224, 3])
        # bizde output 3x14 luk output var 3x14 = 42
        y = tf.placeholder(tf.float32, [None, 224, 224, 14])  # bunu ayarlamalisin
        # outputu reshape ediyorsun
        #y_shaped = tf.reshape(y, [-1, 3, 14])
        # ------burda vgg gectik
        vgg = vgg16(x_shaped, None, sess)  # burda vgg i bos image ile olusturyor
        # # create some convolutional layers
        # layer1 = create_new_conv_layer(x_shaped, 3, 32, [5, 5], [2, 2], name='layer1')
        # layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
        #
        # # flatten the output ready for the fully connected output stage - after two layers of stride 2 pooling, we go
        # # from 28 x 28, to 14 x 14 to 7 x 7 x,y co-ordinates, but with 64 output channels.  To create the fully connected,
        # # "dense" layer, the new shape needs to be [-1, 7 x 7 x 64]
        # flattened = tf.reshape(layer2, [-1, 24 * 24 * 64])
        #
        # # setup some weights and bias values for this layer, then activate with ReLU
        # wd1 = tf.Variable(tf.truncated_normal([24 * 24 * 64, 1000], stddev=0.03), name='wd1')
        # bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
        # dense_layer1 = tf.matmul(flattened, wd1) + bd1
        # dense_layer1 = tf.nn.relu(dense_layer1)
        #
        # # another layer with softmax activations
        # wd2 = tf.Variable(tf.truncated_normal([1000, 42], stddev=0.03), name='wd2')
        # bd2 = tf.Variable(tf.truncated_normal([42], stddev=0.01), name='bd2')
        # dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
        # y_ = tf.nn.softmax(dense_layer2)
        y_ = vgg.probs

        #burda pixelwise sigmoid cross entropy loss function hesapliyacaksin

        # burda digerinin ciktisiyla ve ilk input birbirne ekleyip diger networku cagirmaliyiz
        # burda inputu ayarla

        # _Y ye dokunmayacaksin cunku o zaten heatmaps sorun bizim labellar heatmaps olacak
        # bir sonraki layerlarin inputu
        xHeatmaps = createheatmaps(x_shaped, y_);


        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
        optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

        #-------------------------------------------------

        #bu output labellari icerecek dogru bu  ama input yukardan gelmeli
        y2 = tf.placeholder(tf.float32, [None, 28])

        vgg2 = vgg16Reg(xHeatmaps, None, sess)  # burda vgg i bos image ile olusturyor
        y_2 = vgg2.probs

        cross_entropy2 = tf.reduce_mean(tf.squared_difference(y_2, y2))


        # add an optimiser
        optimiser2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy2)

        # define an accuracy assessment operation
        correct_prediction2 = tf.equal(tf.argmax(y2, 1), tf.argmax(y_2, 1))
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

        # setup the initialisation operator
        init_op = tf.global_variables_initializer()

        # setup recording variables
        # add a summary to store the accuracy
        tf.summary.scalar('accuracy', accuracy2)

        merged = tf.summary.merge_all()

        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(mnist.train.labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                yHeatmaps = createheatmaps2(batch_y)
                _, c, _, c2 = sess.run([optimiser2, cross_entropy2, optimiser, cross_entropy], feed_dict={x: batch_x, y: yHeatmaps, y2: batch_y.reshape(batch_size, 42)[:, :28]})
                avg_cost += c / total_batch
                print("c ", c)
            yHeatmapsTest = createheatmaps2(mnist.test.labels.reshape(len(mnist.test.labels), 42)[:, :28])
            test_acc = sess.run(accuracy2, feed_dict={x: mnist.test.images, y: yHeatmapsTest, y2: mnist.test.labels.reshape(len(mnist.test.labels), 42)[:, :28]})
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
            summary = sess.run(merged, feed_dict={x: mnist.test.images, y: yHeatmapsTest, y2: mnist.test.labels.reshape(len(mnist.test.labels), 42)[:, :28]})
        yHeatmapsLast = createheatmaps2(mnist.test.labels.reshape(len(mnist.test.labels), 42)[:, :28])
        print("\nTraining complete!")
        print(sess.run(accuracy2, feed_dict={x: mnist.test.images, y: yHeatmapsLast, y2: mnist.test.labels.reshape(len(mnist.test.labels), 42)[:, :28]}))


if __name__ == "__main__":
    run_cnn()
