#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import warnings
mnist = input_data.read_data_sets('/temp/data/',one_hot =True)#one_hot in sense of one component is true and rest is off
#we have 10 classes 0-9
# '''
# what actually one-hot mean here
# 0 = [1,0,0,0,0,0,0,0,0,0]#here is one_hot, mean one pixel is one and rest are off,same like other,for 0 class only 0 pixel is on,rest are off
# 1 = [0,1,0,0,0,0,0,0,0,0]
# 2 = [0,0,1,0,0,0,0,0,0,0]
# '''
#prepare our model or computation graph
#we have three hidden layers,because it is deep neural network
nodes_layer1 = 500
nodes_layer2 = 500
nodes_layer3 = 500
#defien our targets
n_classes =10
batch_size =100
#define the batch_size,mean number of training example,or feature we feed to our algorithm
#lets deefine some place holders
#we may mention the placeholder size= height * width
x= tf.placeholder("float",[None,784])#it is not necarry to mention its size,but just for sake if it is our input,it get the input of image in the form of 28*28 pixels
y= tf.placeholder("float")
#now start the feed forward ,we assign the weights and baise,in the key value or dictionary form
def Deep_Neural_Model(data):
    #when we defien a tensorflow variable of random normal it must contain the shape
    hidden_layer1 = {'weights':tf.Variable(tf.random_normal([784,nodes_layer1])),
                    'baise':tf.Variable(tf.random_normal([nodes_layer1]))}
    hidden_layer2 = {'weights':tf.Variable(tf.random_normal([nodes_layer1,nodes_layer2])),
                    'baise':tf.Variable(tf.random_normal([nodes_layer2]))}
    hidden_layer3 = {'weights':tf.Variable(tf.random_normal([nodes_layer2,nodes_layer3])),
                    'baise':tf.Variable(tf.random_normal([nodes_layer3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([nodes_layer3,n_classes])),
                    'baise':tf.Variable(tf.random_normal([n_classes]))}
    #baise?  (input * weight ) +baise (the role of baise is if the weights and input neuron are zero,the some neuron should fire due to baise)
#   (input * weight ) +baise.. this is actually our model,that we design for each layer , let sdeign ou model
    
    l1 =tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['baise'])
    #apply the activation function
    l1 = tf.nn.relu(l1)#here nn is the neural network operations
    l2 =tf.add(tf.linalg.matmul(l1,hidden_layer2['weights']), hidden_layer2['baise'])
    l2 = tf.nn.relu(l2)
    l3 =tf.add(tf.linalg.matmul(l2,hidden_layer3['weights']), hidden_layer3['baise'])
    l3 = tf.nn.relu(l3)
    output =tf.matmul(l3,output_layer['weights'] )+ output_layer['baise']
    return output#this output is the one-hot array
#at this stage we complete our computation graph

#now train our model
def train_deep_neural_model(x):#it take just the input here the x
    #make predictions
    prediction = Deep_Neural_Model(x)
    #calculate the cost or in other word loss or error .cost = target - predicted
    #tf.nn.softmax_cross_entropy_with_logits(this function is used to calculate the difference of predicted- target)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    #now our objective is that to minimize that cost,for this we use an optimizer,here we use the AdamOptimizer instead of gradient descen
    optimizer = tf.train.AdamOptimizer().minimize(cost)#this optimizer take optional parameter which is learning rate,it is by default 0.001,so we cannot modify it
    hm_epoch =10 #we know epoch =forward feed + backpropogation,actually they are cycles
    #run the seesion for computational graph
    init = tf.global_variables_initializer()
    with tf.Session() as s:
        init.run()
        for epoch in range(hm_epoch):
            epoch_lose = 0#at initially
            #now we need to find how many cycle we need on the base of total smaple and batch_size that we define above
            for _ in range(int(mnist.train.num_examples/batch_size)):
                batch_x,batch_y =mnist.train.next_batch(batch_size)
                _,c = s.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})#Use feed_dict To Feed Values To TensorFlow Placeholders
                epoch_lose+=c
            print('Epoch=',epoch,'Completed out of = ',hm_epoch,'lose = ',epoch_lose)
        #to get the maximum value of prediction index and the actual label index it should be same
        correct = tf.equal(tf.arg_max(prediction,1),tf.arg_max(y,1))
        #calculate the accuracy of the model
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))#reduce mean calculte the mean,and tf.cast change the variable into the float
        print('Accuracy = ',accuracy.eval({x:mnist.test.images , y:mnist.test.labels}))
    
    
    


train_deep_neural_model(x)


# In[ ]:




