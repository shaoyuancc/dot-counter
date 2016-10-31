import tensorflow as tf
import dot_input_manager as dim 
import matplotlib.pyplot as plt
import numpy
from scipy import misc

n_classes = 10
batch_size = 128
height = 28
width = 28
n_pixels = height*width

### CHANGE THIS TO SUIT YOUR SYSTEM
dotcounterdir = '/home/shao/Documents/DotCounter/'

n_pixels = height*width

x = tf.placeholder('float', [None, int(n_pixels)])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):

    ### VERY IMPORTANT MUST MATCH STRUCTURE OF NET USED TO CREATE THE MODEL
    weights = {'W_conv1':tf.Variable(tf.random_normal([2,2,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([2,2,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def implement_neural_network(image_array):
    prediction = convolutional_neural_network(x)
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    #optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #hm_epochs = 10
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, (dotcounterdir + 'saved_nets/dots_net.ckpt'))
        result_array = sess.run(prediction, feed_dict={x: image_array})
        print(result_array)
        result = numpy.argmax(result_array, axis = 1)

        print('The Predicted Result is: ', result)
        #plt.imshow(x[0,:,:,0])
        #print(labels_array[selected_index])
        #plt.gray()
        #plt.show()
        

def print_answer(response):
#image_tensor = tf.to_float(tf.read_file(response))

    image_array = numpy.zeros([1,28,28],dtype=numpy.float32)
    image_array[0] = misc.imread(response)
    print(image_array.shape)
#image_tensor = tf.convert_to_tensor(image_array[0],dtype=tf.float32)
    image_array = image_array[:,:,:,numpy.newaxis]
    print(image_array.shape)
    assert image_array.shape[3] == 1
    print(image_array.shape)
    image_array = image_array.reshape(image_array.shape[0],
                        height*width) 
    print(image_array.shape)
    image_array = image_array.astype(numpy.float32)
    image_array = numpy.multiply(image_array, 1.0 / 255.0)
    print(image_array.shape)

#image_array = tf.convert_to_tensor(image_array, dtype=tf.float32)
    implement_neural_network(image_array)

response = input('Please Enter Filepath of image enclosed in single inverted commas:')
#if response.endswith(".jpg'"):
response = response.split("'")
print(response[1])
print_answer(response[1])

