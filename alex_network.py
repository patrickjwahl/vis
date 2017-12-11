import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 15
load_size = 256
fine_size = 224
c = 1
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 50000
step_display = 50
step_save = 10000
path_save = 'alexnet_bn'
start_from = ''

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)
    
def alexnet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
        'wc6': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=np.sqrt(2./(3*3*128)))),
        'wc7': tf.Variable(tf.random_normal([5, 5, 96, 128], stddev=np.sqrt(2./(3*3*98)))),
        'wc8a': tf.Variable(tf.random_normal([11, 11, 50, 96], stddev=np.sqrt(2./(3*3*50)))),
        'wc8b': tf.Variable(tf.random_normal([11, 11, 50, 96], stddev=np.sqrt(2./(3*3*50)))),
        'wc8c': tf.Variable(tf.random_normal([11, 11, 50, 96], stddev=np.sqrt(2./(3*3*50)))),

        'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'ba': tf.Variable(tf.ones([224, 224, 50])),
        'bb': tf.Variable(tf.ones([224, 224, 50])),
        'bc': tf.Variable(tf.ones([224, 224, 50]))
    }

    # Conv + ReLU + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    #7->28
    conv6 = tf.nn.conv2d_transpose(pool5, weights['wc6'], output_shape=[batch_size, 28, 28, 128], strides=[1, 4, 4, 1], padding='SAME')
    conv6 =  batch_norm_layer(conv6, train_phase, 'bn6')
    conv6 = tf.nn.relu(conv6)
    pool6 = tf.nn.max_pool(conv6, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    #28->112
    conv7 = tf.nn.conv2d_transpose(pool6, weights['wc7'], output_shape=[batch_size, 112, 112, 96], strides=[1, 4, 4, 1], padding='SAME')
    conv7 =  batch_norm_layer(conv7, train_phase, 'bn7')
    conv7 = tf.nn.relu(conv7)
    pool7 = tf.nn.max_pool(conv7, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    #112->224
    conv8a = tf.nn.conv2d_transpose(pool7, weights['wc8a'], output_shape=[batch_size, 224, 224, 50], strides=[1, 2, 2, 1])
    conv8b = tf.nn.conv2d_transpose(pool7, weights['wc8b'], output_shape=[batch_size, 224, 224, 50], strides=[1, 2, 2, 1])
    conv8c = tf.nn.conv2d_transpose(pool7, weights['wc8c'], output_shape=[batch_size, 224, 224, 50], strides=[1, 2, 2, 1])

    conv8a = tf.nn.dropout(conv8a, keep_dropout)
    conv8b = tf.nn.dropout(conv8b, keep_dropout)
    conv8c = tf.nn.dropout(conv8c, keep_dropout)

    # Output FC
    outA, outB, outC = tf.add(conv8a, biases['ba']), tf.add(conv8b, biases['bb']), tf.add(conv8c, biases['bc'])

    vars = 0
    for v in tf.global_variables():
        vars += np.prod(v.get_shape().as_list())

    print vars 
    
    return outA, outB, outC

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True,
    'test': False,
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False,
    'test': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [batch_size, fine_size, fine_size, c])

def binner(image):
    return tf.cast(tf.floor(tf.scalar_mul(tf.constant(49.9999, dtype=tf.float32), image)), tf.int64)

y = tf.placeholder(tf.float32, [batch_size, fine_size, fine_size, 3])
actual_y = binner(y)

keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logitsR, logitsG, logitsB = alexnet(x, keep_dropout, train_phase)

# Define loss and optimizer
# Define loss and optimizer
loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(tf.slice(actual_y, [0,0,0,0], [batch_size, fine_size, fine_size, 1])), logits=logitsR))
loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(tf.slice(actual_y, [0,0,0,1], [batch_size, fine_size, fine_size, 1])), logits=logitsG))
loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(tf.slice(actual_y, [0,0,0,2], [batch_size, fine_size, fine_size, 1])), logits=logitsB))
loss = loss1+loss2+loss3
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

# Launch the graph
with tf.Session(config=config) as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)
    
    step = 0

    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        
        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l = sess.run(loss, feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Training Loss= " + "{0}".format(l))

            # stat_file = open('stat_file.txt', 'a')
            # stat_file.write("{:.6f}\n".format(l))
            # stat_file.close()
        
        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x:images_batch,y:labels_batch,keep_dropout:dropout,train_phase:True})
        
        step += 1
        
        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))
        
    print("Optimization Finished!")