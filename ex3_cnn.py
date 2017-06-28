# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:36:05 2017

@author: Alphatao
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#载入数据 创建sess
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()

#初始化权重，偏置函数 可重复使用
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#定义卷积层池化层创建函数
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#输入数据
with tf.name_scope("input"):    
    x = tf.placeholder(tf.float32, [None, 784]) #输入数据
    y_ = tf.placeholder(tf.float32, [None, 10]) #对应label
    x_image = tf.reshape(x, [-1, 28, 28, 1])

#第一层
with tf.name_scope("layer1"):
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    tf.summary.histogram("weight",W_conv1)
    tf.summary.histogram("bias",b_conv1)

#第二层
with tf.name_scope("layer2"):
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    tf.summary.histogram("weight",W_conv2)
    tf.summary.histogram("bias",b_conv2)

#全连接层1 重定义上一层输出的大小，dropout
with tf.name_scope("layer_fc"):
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    tf.summary.histogram("weight",W_fc1)
    tf.summary.histogram("bias",b_fc1)

#softmax层
with tf.name_scope("layer_sf"):
    W_sf = weight_variable([1024, 10])
    b_sf = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_sf) + b_sf)
    tf.summary.histogram("weight",W_sf)
    tf.summary.histogram("bias",b_sf)

#损失函数
with tf.name_scope('loss'):
    cross = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1]))
    tf.summary.scalar('loss',cross)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross)

#准确率
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('acc',accuracy)

merged = tf.summary.merge_all() 
writer = tf.summary.FileWriter("logs/",sess.graph)  

#开始训练
tf.global_variables_initializer().run()
for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step%d, train accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    if i%10==0:
        result = sess.run(merged,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5}) 
        writer.add_summary(result,i)

print("test accuracy %g "%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))



