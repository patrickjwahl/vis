import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 3
load_size = 331
fine_size = 299
c = 1
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate_initial = 0.0001
learning_rate_decay = 0.94
decay_steps = int(200000/batch_size)
#momentum = 0.9
dropout = 0.5 # Dropout, probability to keep units
training_iters = 50000
step_display = 50
step_save = 5000
path_save = 'vis'
start_from = ''

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)
    
def fun_net(x, keep_dropout, train_phase):
    weights = {

        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=np.sqrt(2./(3*3*1)))),
        'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=np.sqrt(2./(3*3*32)))),

        'wsc3d': tf.Variable(tf.random_normal([3, 3, 64, 1], stddev=np.sqrt(2./(3*3*64)))),
        'wsc3p': tf.Variable(tf.random_normal([1, 1, 64, 128], stddev=np.sqrt(2./(1*1*64)))),

        'wsc4d': tf.Variable(tf.random_normal([3, 3, 128, 1], stddev=np.sqrt(2./(3*3*128)))),
        'wsc4p': tf.Variable(tf.random_normal([1, 1, 128, 128], stddev=np.sqrt(2./(1*1*128)))),

        'wc4a': tf.Variable(tf.random_normal([1, 1, 64, 128], stddev=np.sqrt(2./(1*1*64)))),

        'wsc5d': tf.Variable(tf.random_normal([3, 3, 128, 1], stddev=np.sqrt(2./(3*3*128)))),
        'wsc5p': tf.Variable(tf.random_normal([1, 1, 128, 256], stddev=np.sqrt(2./(1*1*128)))),

        'wsc6d': tf.Variable(tf.random_normal([3, 3, 256, 1], stddev=np.sqrt(2./(3*3*256)))),
        'wsc6p': tf.Variable(tf.random_normal([1, 1, 256, 256], stddev=np.sqrt(2./(1*1*256)))),

        'wc6a': tf.Variable(tf.random_normal([1, 1, 128, 256], stddev=np.sqrt(2./(1*1*128)))),

        'wsc7d': tf.Variable(tf.random_normal([3, 3, 256, 1], stddev=np.sqrt(2./(3*3*256)))),
        'wsc7p': tf.Variable(tf.random_normal([1, 1, 256, 728], stddev=np.sqrt(2./(1*1*256)))),

        'wsc8d': tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728)))),
        'wsc8p': tf.Variable(tf.random_normal([1, 1, 728, 728], stddev=np.sqrt(2./(1*1*728)))),

        'wc8a': tf.Variable(tf.random_normal([1, 1, 256, 728], stddev=np.sqrt(2./(1*1*256)))),

        'wsc9d': tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728)))),
        'wsc9p': tf.Variable(tf.random_normal([1, 1, 728, 728], stddev=np.sqrt(2./(1*1*728)))),

        'wsc10d': tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728)))),
        'wsc10p': tf.Variable(tf.random_normal([1, 1, 728, 1024], stddev=np.sqrt(2./(1*1*728)))),

        'wc10a': tf.Variable(tf.random_normal([1, 1, 728, 1024], stddev=np.sqrt(2./(1*1*728)))),

        'wsc11d': tf.Variable(tf.random_normal([3, 3, 1024, 1], stddev=np.sqrt(2./(3*3*1024)))),
        'wsc11p': tf.Variable(tf.random_normal([1, 1, 1024, 1536], stddev=np.sqrt(2./(1*1*1024)))),

        'wsc12d': tf.Variable(tf.random_normal([3, 3, 1536, 1], stddev=np.sqrt(2./(3*3*1536)))),
        'wsc12p': tf.Variable(tf.random_normal([1, 1, 1536, 2048], stddev=np.sqrt(2./(1*1*1536)))),

        'wct13': tf.Variable(tf.random_normal([3, 3, 1024, 2048], stddev=np.sqrt(2./(1*1*2048)))),
        'wct14': tf.Variable(tf.random_normal([3, 3, 512, 1024], stddev=np.sqrt(2./(1*1*1024)))),
        'wct15': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=np.sqrt(2./(1*1*512)))),

        'wct16a': tf.Variable(tf.random_normal([2, 2, 256, 50], stddev=np.sqrt(2./(1*1*256)))),
        'wct16b': tf.Variable(tf.random_normal([2, 2, 256, 50], stddev=np.sqrt(2./(1*1*256)))),
        'wct16c': tf.Variable(tf.random_normal([2, 2, 256, 50], stddev=np.sqrt(2./(1*1*256))))

    }

    biases = {
        'ba': tf.Variable(tf.ones([299, 299, 50])),
        'bb': tf.Variable(tf.ones([299, 299, 50])),
        'bc': tf.Variable(tf.ones([299, 299, 50]))
    }


    # Middle flow weights #
    for i in range(8):
        depthwise1 = 'mwsc{0}d'.format(i*3+1)
        depthwise2 = 'mwsc{0}d'.format(i*3+2)
        depthwise3 = 'mwsc{0}d'.format(i*3+3)

        pointwise1 = 'mwsc{0}p'.format(i*3+1)
        pointwise2 = 'mwsc{0}p'.format(i*3+2)
        pointwise3 = 'mwsc{0}p'.format(i*3+3)

        weights[depthwise1] = tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728))))
        weights[depthwise2] = tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728))))
        weights[depthwise3] = tf.Variable(tf.random_normal([3, 3, 728, 1], stddev=np.sqrt(2./(3*3*728))))

        weights[pointwise1] = tf.Variable(tf.random_normal([1, 1, 728, 728], stddev=np.sqrt(2./(1*1*728))))
        weights[pointwise2] = tf.Variable(tf.random_normal([1, 1, 728, 728], stddev=np.sqrt(2./(1*1*728))))
        weights[pointwise3] = tf.Variable(tf.random_normal([1, 1, 728, 728], stddev=np.sqrt(2./(1*1*728))))


    # Entry flow ====================================================================================================================

    #Layer 1: conv32, 3x3, stride 2x2, relu
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)

    #Layer 2: conv64, 3x3, relu
    conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)

    #Layer 3: Sep Conv 128, 3x3, relu
    conv3 = tf.nn.separable_conv2d(conv2, weights['wsc3d'], weights['wsc3p'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    #Layer 4: Sep Conv 128, 3x3, max pool, 3x3, stride 2x2
    conv4 = tf.nn.separable_conv2d(conv3, weights['wsc4d'], weights['wsc4p'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    #Layer 4a: conv128, 1x1, stride 2x2
    conv4a = tf.nn.conv2d(conv2, weights['wc4a'], strides=[1, 2, 2, 1], padding='SAME')
    conv4a = batch_norm_layer(conv4a, train_phase, 'bn4a')
    conv4 = tf.add(conv4, conv4a)

    #Layer 5: relu, sep conv 256, 3x3
    conv5 = tf.nn.relu(conv4);
    conv5 = tf.nn.separable_conv2d(conv5, weights['wsc5d'], weights['wsc5p'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')

    #Layer 6: relu, sep conv 256, 3x3, max pool
    conv6 = tf.nn.relu(conv5)
    conv6 = tf.nn.separable_conv2d(conv6, weights['wsc6d'], weights['wsc6p'], strides=[1, 1, 1, 1], padding='SAME')
    conv6 = batch_norm_layer(conv6, train_phase, 'bn6')
    conv6 = tf.nn.max_pool(conv6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    #Layer 6a: conv 256, 1x1, stride 2x2
    conv6a = tf.nn.conv2d(conv4a, weights['wc6a'], strides=[1, 2, 2, 1], padding='SAME')
    conv6a = batch_norm_layer(conv6a, train_phase, 'bn6a')
    conv6 = tf.add(conv6, conv6a)

    #Layer 7: relu, sep conv 728, 3x3
    conv7 = tf.nn.relu(conv6);
    conv7 = tf.nn.separable_conv2d(conv7, weights['wsc7d'], weights['wsc7p'], strides=[1, 1, 1, 1], padding='SAME')
    conv7 = batch_norm_layer(conv7, train_phase, 'bn7')

    #Layer 8: relu, sep conv 728, 3x3, max pool
    conv8 = tf.nn.relu(conv7);
    conv8 = tf.nn.separable_conv2d(conv8, weights['wsc8d'], weights['wsc8p'], strides=[1, 1, 1, 1], padding='SAME')
    conv8 = batch_norm_layer(conv8, train_phase, 'bn8')
    conv8 = tf.nn.max_pool(conv8, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    #Layer 8a: conv 728, 1x1, stride 2x2
    conv8a = tf.nn.conv2d(conv6a, weights['wc8a'], strides=[1, 2, 2, 1], padding='SAME')
    entry_flow_out = tf.add(conv8, conv8a)



    # Middle flow --------------------------------------------------------------------------------------------------------------

    middle_flow_val = entry_flow_out

    for i in range(8):

        stored_value = middle_flow_val

        middle_flow_val = tf.nn.relu(middle_flow_val)
        middle_flow_val = tf.nn.separable_conv2d(middle_flow_val, weights['mwsc{0}d'.format(i*3+1)], weights['mwsc{0}p'.format(i*3+1)], strides=[1, 1, 1, 1], padding='SAME')
        middle_flow_val = batch_norm_layer(middle_flow_val, train_phase, 'mbn{0}'.format(i*3+1))

        middle_flow_val = tf.nn.relu(middle_flow_val)
        middle_flow_val = tf.nn.separable_conv2d(middle_flow_val, weights['mwsc{0}d'.format(i*3+2)], weights['mwsc{0}p'.format(i*3+2)], strides=[1, 1, 1, 1], padding='SAME')
        middle_flow_val = batch_norm_layer(middle_flow_val, train_phase, 'mbn{0}'.format(i*3+2))

        middle_flow_val = tf.nn.relu(middle_flow_val)
        middle_flow_val = tf.nn.separable_conv2d(middle_flow_val, weights['mwsc{0}d'.format(i*3+3)], weights['mwsc{0}p'.format(i*3+3)], strides=[1, 1, 1, 1], padding='SAME')
        middle_flow_val = batch_norm_layer(middle_flow_val, train_phase, 'mbn{0}'.format(i*3+3))

        middle_flow_val = tf.add(middle_flow_val, stored_value)




    # Exit flow -------------------------------------------------------------------------------------------------------------------

    conv9 = tf.nn.relu(middle_flow_val)
    conv9 = tf.nn.separable_conv2d(conv9, weights['wsc9d'], weights['wsc9p'], strides=[1, 1, 1, 1], padding='SAME')
    conv9 = batch_norm_layer(conv9, train_phase, 'bn9')

    conv10 = tf.nn.relu(conv9)
    conv10 = tf.nn.separable_conv2d(conv10, weights['wsc10d'], weights['wsc10p'], strides=[1, 1, 1, 1], padding='SAME')
    conv10 = batch_norm_layer(conv10, train_phase, 'bn10')
    conv10 = tf.nn.max_pool(conv10, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv10a = tf.nn.conv2d(middle_flow_val, weights['wc10a'], strides=[1, 2, 2, 1], padding='SAME')
    conv10a = batch_norm_layer(conv10a, train_phase, 'bn10a')
    conv10 = tf.add(conv10, conv10a)

    conv11 = tf.nn.separable_conv2d(conv10, weights['wsc11d'], weights['wsc11p'], strides=[1, 1, 1, 1], padding='SAME')
    conv11 = batch_norm_layer(conv11, train_phase, 'bn11')
    conv11 = tf.nn.relu(conv11)

    conv12 = tf.nn.separable_conv2d(conv11, weights['wsc12d'], weights['wsc12p'], strides=[1, 1, 1, 1], padding='SAME')
    conv12 = batch_norm_layer(conv12, train_phase, 'bn12')
    conv12 = tf.nn.relu(conv12)

    #Transpose convs start

    conv13 = tf.nn.conv2d_transpose(conv12, weights['wct13'], output_shape=[batch_size, 50, 50, 1024], strides=[1, 5, 5, 1], padding='SAME')
    conv13 = batch_norm_layer(conv13, train_phase, 'bn13')
    conv13 = tf.nn.relu(conv13)
    conv14 = tf.nn.conv2d_transpose(conv13, weights['wct14'], output_shape=[batch_size, 150, 150, 512], strides=[1, 3, 3, 1], padding='SAME')
    conv14 = batch_norm_layer(conv14, train_phase, 'bn14')
    conv14 = tf.nn.relu(conv14)
    conv15 = tf.nn.conv2d_transpose(conv14, weights['wct15'], output_shape=[batch_size, 300, 300, 256], strides=[1, 2, 2, 1], padding='SAME')
    conv15 = batch_norm_layer(conv15, train_phase, 'bn15')
    conv15 = tf.nn.relu(conv15)
    conv16a = tf.nn.conv2d(conv15, weights['wct16a'], strides=[1, 1, 1, 1], padding='VALID')
    conv16b = tf.nn.conv2d(conv15, weights['wct16b'], strides=[1, 1, 1, 1], padding='VALID')
    conv16c = tf.nn.conv2d(conv15, weights['wct16c'], strides=[1, 1, 1, 1], padding='VALID')

    conv17a = tf.nn.dropout(conv16a, keep_dropout)
    conv17b = tf.nn.dropout(conv16b, keep_dropout)
    conv17c = tf.nn.dropout(conv16c, keep_dropout)

    outA, outB, outC = tf.add(conv17a, biases['ba']), tf.add(conv17b, biases['bb']), tf.add(conv17c, biases['bc'])

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
    'test': False
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
logitsR, logitsG, logitsB = fun_net(x, keep_dropout, train_phase)

# Define learning rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate_initial, global_step, decay_steps, learning_rate_decay, staircase=True)

# Define loss and optimizer
loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(tf.slice(actual_y, [0,0,0,0], [batch_size, fine_size, fine_size, 1])), logits=logitsR))
loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(tf.slice(actual_y, [0,0,0,1], [batch_size, fine_size, fine_size, 1])), logits=logitsG))
loss3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(tf.slice(actual_y, [0,0,0,2], [batch_size, fine_size, fine_size, 1])), logits=logitsB))
loss = tf.add(loss1, tf.add(loss2, loss3))
train_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'

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