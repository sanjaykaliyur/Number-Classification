
# coding: utf-8

# # Deep learning for image classification
# 
# This Python notebook explores [deep learning](https://en.wikipedia.org/wiki/Deep_learning)  for neural networks and creates a classification model that maps images of single digit numbers to their corresponding numeric representations.

# ## Load libraries

# Install the [nolearn](https://pythonhosted.org/nolearn/) deep learning Python library. 
# 
# This is the main library used for all of our deep learning purposes.

# In[1]:

get_ipython().system(u'pip install --user nolearn')


# Import libraries 

# In[2]:

import warnings
warnings.filterwarnings("ignore")
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic(u'matplotlib inline')


# ## Load data

# The data set for this classification model comes from the MNIST database. It is a large database of handwritten numbers that uses real-world data.

# In[3]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")


# Training set images and labels.

# In[4]:

mnist_images = mnist.train.images
mnist_labels = mnist.train.labels


# Split the data into training and testing data sets.
# 
# trX = Training set images, trY = Training set labels, teX = Testing set images, teY = Testing set labels

# In[5]:

trX, trY, teX, teY = train_test_split(mnist_images, mnist_labels.astype("int0"), test_size = 0.33)


# Output number of images in each data set.

# In[6]:

print "Number of images for training:", trX.shape[0]
print "Number of images used for testing:", teX.shape[0]


# ## Train the classification model for pattern recognition

# Uses the Deep Belief Network solves to train a pattern recognition model for handwritten numbers.
# 
# The most important values to notice when creating a Deep Belief Network are the learning rate and the epochs. 

# The [learning rate](https://www.coursera.org/learn/machine-learning/lecture/3iawu/gradient-descent-in-practice-ii-learning-rate) dictates the length of the steps the algorithm will take during [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent). The lower the value of the learning rate, the more steps the algorithm will take. Generally, the lower the value, the longer the model will take to run.
# 
# [Epochs](https://stackoverflow.com/questions/31155388/meaning-of-an-epoch-in-neural-networks-training) are another word for the number of times an algorithm iterates over the training data set. Generally, the more epochs, the more accurate the model. More epochs also means the model will take longer to run.

# In this initial version of our Deep Belief Network, we will use 10 epochs and set the learning rate to 0.3. We can modify these values in the future and determine which combination gives us the most accurate model.

# In[26]:

dbn = DBN(
    [trX.shape[1], 300, 10], 
    learn_rates = 0.1,
    learn_rate_decays = 0.9,
    epochs = 10,
    verbose = 1)
dbn.fit(trX, teX)


# Notice that through each iteration the loss and error decreases.

# ## Testing the model – Version 1

# Performance of the Classification Model:

# In[27]:

predictions = dbn.predict(trY)
print classification_report(teY, predictions)
print 'The accuracy is:', accuracy_score(teY, predictions)


# This accuracy is quite good for our first attempt. Notice that the model performs the best when attempting to identify the number '1' and the worst when attempting to identify the number '9'. 

# Now to see our model in action. We need to create a function that generates a random image of a number and feeds it to the classification model for identification. When called, an image will appear on screen along with the model's guess for what digit is being represented. Each function call will produce a randomized image.

# In[9]:

def displayImage():
    i = np.random.choice(np.arange(0, len(teY)), size = (1,))
    pred = dbn.predict(np.atleast_2d(trY[i]))
    image = (trY[i] * 255).reshape((28, 28)).astype("uint8")
    imgplot = plt.imshow(image)
    imgplot.set_cmap('gray')
    print "Actual digit:", teY[i][0]
    print "Classification Model prediction:", pred[0]


# Function call

# In[10]:

displayImage()


# In[11]:

displayImage()


# In[12]:

displayImage()


# In[13]:

displayImage()


# ## Improving the classification model – Version 2: Deep Belief Network modifications

# Even though the initial model has close to 98% accuracy, we can always make our model more accurate. One potential way of increasing the accuracy of the classification model is to make modifications to the way we train the Deep Belief Network. 

# To make the model more accurate, I decreased the learning rate from 0.3 to 0.2 in order to allow for smaller steps during Gradient Descent. I also increased the number of epochs from 10 to 50.

# In[14]:

dbn = DBN(
    [trX.shape[1], 300, 10], 
    learn_rates = 0.2, # Modified Field
    learn_rate_decays = 0.9,
    epochs = 50, # Modified Field
    verbose = 1)
dbn.fit(trX, teX)


# Notice: In general, the loss and error both decrease over time.

# ## Testing the model – Version 2

# Performance of the updated Classification Model:

# In[15]:

predictions = dbn.predict(trY)
print classification_report(teY, predictions)
print 'The accuracy is:', accuracy_score(teY, predictions)


# The model has gained about 0.5% accuracy by making these small changes to the Deep Belief Network. Notice that the model still identifies the number '1' approximately as well as the initial model. The number '9', which the initial model had the toughest time identifying, has seen an approximately 2% increase in precision.

# In[16]:

displayImage()


# In[17]:

displayImage()


# In[18]:

displayImage()


# In[28]:

displayImage()


# ## Improving the classification model – Version 3: Dropout

# [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) regularization is a technique used for Neural Networks and Deep Learning models that is aimed at reducing [overfitted](https://en.wikipedia.org/wiki/Overfitting) models. 
# 
# Overfitting is a symptom of a poorly-performing model. The previous versions of our model were very successful so it is unlikely that they overfit data in any significant manner, but it is worth experimenting with the dropout rate to see if our model will perform better. 

# To change the dropout rate, we will need to include dropouts in our Deep Belief Network. As a test, we will set dropouts to 0.1.

# In[20]:

dbn = DBN(
    [trX.shape[1], 300, 10], 
    learn_rates = 0.2,
    learn_rate_decays = 0.9,
    dropouts = 0.2, # Added Field
    epochs = 50,
    verbose = 1)
dbn.fit(trX, teX)


# ## Testing the model – Version 3

# In[21]:

predictions = dbn.predict(trY)
print classification_report(teY, predictions)
print 'The accuracy is:', accuracy_score(teY, predictions)


# This is the highest accuracy that we have observed. Accounting for overfitting has resulted in an approximately 0.3% increase in accuracy. 
# 
# Notice that the number '9' is no longer the most difficult number for our model to identify and has instead been replaced by the number '8'.

# In[22]:

displayImage()


# In[23]:

displayImage()


# In[24]:

displayImage()


# In[25]:

displayImage()


# ## Summary

# Through a few iterations of our model, we have nearly achieved perfect accuracy.

# In general, deep learning works best when looking at several terabytes of data. Processing this amount of data is essentially impossible on a single computer and thus requires the use of a computer with several GPUs or some cloud computing technology (IBM Cloud Compute, Google Cloud Platform, etc).

# The best way to improve the model's performance is to use more computing power. This comes in the form of using a more powerful computer with at least one GPU. Another option is to move all computation to the cloud. By doing these things, you can increase how fast the models work. This will allow you to use a much larger data set to train the algorithm. The more data, the more accurate the model. 

# In[ ]:



