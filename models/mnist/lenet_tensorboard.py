import os
import shutil
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


log_dir = '/tmp/tensorflow/logs/MNIST_with_summaries'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

### LeNet
# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# NN
def nn_layer(input_tensor, input_shape, output_shape, layer_name,
             cnn=False, act=tf.nn.relu, pooling=False):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = weight_variable(input_shape)
        with tf.name_scope('biases'):
            b = bias_variable(output_shape)
        if cnn:
            with tf.name_scope('convolution'):
                h = act(conv2d(input_tensor, W) + b)
        else:
            with tf.name_scope('activation'):
                h = act(tf.matmul(input_tensor, W) + b)
    if pooling:
        with tf.name_scope('pooling_layer'):
            h_pool = max_pool_2x2(h)
        return h_pool, W, b
    else:
        return h, W, b

# build tensors
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
with tf.name_scope('input_image'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 6)
    
# First Convolutional Layer
layer1_num_conv = 32
layer1_input_shape = [5, 5, 1, layer1_num_conv]
layer1_output_shape = [layer1_num_conv]
layer1, W1, b1 = nn_layer(x_image, layer1_input_shape, layer1_output_shape,
                          cnn=True, pooling=True,
                          layer_name='First_Convolutional_Layer')

# Second Convolutional Layer
layer2_num_conv = 64
layer2_input_shape = [5, 5, layer1_num_conv, layer2_num_conv]
layer2_output_shape = [layer2_num_conv]
layer2, W2, b2 = nn_layer(layer1, layer2_input_shape, layer2_output_shape,
                          cnn=True, pooling=True,
                          layer_name='Second_Convolutional_Layer')

# Densely Connected Layer
W_fc1_shape = [7 * 7 * layer2_num_conv, 1024]
b_fc1_shape = [1024]
layer2_flat = tf.reshape(layer2, [-1, 7 * 7 * layer2_num_conv])
h_fc1, W_fc1, _ = nn_layer(layer2_flat, W_fc1_shape, b_fc1_shape,
                           layer_name='Densely_Connected_Layer')

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2_shape = [1024, 10]
b_fc2_shape = [10]
y_conv, W_fc2, _ = nn_layer(h_fc1_drop, W_fc2_shape, b_fc2_shape,
                            layer_name='Readout_Layer')

# Train and Evaluate the Model
with tf.name_scope('loss_function'):
    with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar('loss_function', loss)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# initial session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# merge tf.summary
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))
saver = tf.train.Saver()

def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(50)
        dropout = 0.5
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

# running and logging
# Train the model, and also write summaries.
# Every 100th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data, & add training summaries
max_epoch = 10000
for i in range(max_epoch):
    if i % 100 == 0:
        test_summary, test_acc = sess.run([merged, accuracy], feed_dict=feed_dict(train=False))
        test_writer.add_summary(test_summary, i)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        train_summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(train=True))
        train_writer.add_run_metadata(run_metadata, 'step{0}'.format(i))
        train_writer.add_summary(train_summary, i)
        print("step {0}, testing accuracy {1:.4f}".format(i, test_acc))
    else:
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(train=True))
        train_writer.add_summary(summary, i)
    if i % 1000 == 0:
        saver.save(sess, os.path.join(log_dir, 'model-step-{0}.ckpt'.format(i)), i)

test_summary, test_acc = sess.run([merged, accuracy],
                                  feed_dict=feed_dict(False))
test_writer.add_summary(test_summary, i)
print("test accuracy {0}".format(test_acc))

sess.close()
train_writer.close()
test_writer.close()
