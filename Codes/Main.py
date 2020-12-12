#!/usr/bin/env python
# coding: utf-8

# STEP 0: PROBLEM STATEMENT
# In this case study, you have been provided with images of traffic signs and the goal is to train a Deep Network to classify them
# 
# The dataset contains 43 different classes of images.
# 
# Classes are as listed below:
# 
# ( 0, b'Speed limit (20km/h)') 
# 
# ( 1, b'Speed limit (30km/h)')
# 
# ( 2, b'Speed limit (50km/h)') 
# 
# ( 3, b'Speed limit (60km/h)')
# 
# ( 4, b'Speed limit (70km/h)') 
# 
# ( 5, b'Speed limit (80km/h)')
# 
# ( 6, b'End of speed limit (80km/h)') 
# 
# ( 7, b'Speed limit (100km/h)')
# 
# ( 8, b'Speed limit (120km/h)') 
# 
# ( 9, b'No passing')
# 
# (10, b'No passing for vehicles over 3.5 metric tons')
# 
# (11, b'Right-of-way at the next intersection') 
# 
# (12, b'Priority road')
# 
# (13, b'Yield') (14, b'Stop') 
# 
# (15, b'No vehicles')
# 
# (16, b'Vehicles over 3.5 metric tons prohibited') 
# 
# (17, b'No entry')
# 
# (18, b'General caution') 
# 
# (19, b'Dangerous curve to the left')
# 
# (20, b'Dangerous curve to the right') 
# 
# (21, b'Double curve')
# 
# (22, b'Bumpy road') 
# 
# (23, b'Slippery road')
# 
# (24, b'Road narrows on the right') 
# 
# (25, b'Road work')
# 
# (26, b'Traffic signals') 
# 
# (27, b'Pedestrians') 
# 
# (28, b'Children crossing')
# 
# (29, b'Bicycles crossing') 
# 
# (30, b'Beware of ice/snow')
# 
# (31, b'Wild animals crossing')
# 
# (32, b'End of all speed and passing limits') 
# 
# (33, b'Turn right ahead')
# 
# (34, b'Turn left ahead') 
# 
# (35, b'Ahead only') 
# 
# (36, b'Go straight or right')
# 
# (37, b'Go straight or left') 
# 
# (38, b'Keep right') 
# 
# (39, b'Keep left')
# 
# (40, b'Roundabout mandatory') 
# 
# (41, b'End of no passing')
# 
# (42, b'End of no passing by vehicles over 3.5 metric tons')
# 

# In[1]:


# Module Imported
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import cv2


# In[2]:


# Reading Dataset
with open("./traffic-signs-data/train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("./traffic-signs-data/valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("./traffic-signs-data/test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)


# In[3]:


# Initialize
X_train,y_train = train['features'],train['labels']
X_validation , y_validation = valid['features'],valid['labels']
X_test , y_test = test['features'],test['labels']


# In[4]:


X_train.shape


# In[5]:


X_test.shape


# # Image exploration

# In[6]:


i =102
plt.imshow(X_train[i])
y_train[i]


# # Data preparation

# In[7]:


from sklearn.utils import shuffle
X_train,y_train = shuffle(X_train,y_train)


# In[8]:


X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_test_gray  = np.sum(X_test/3, axis=3, keepdims=True)
X_validation_gray  = np.sum(X_validation/3, axis=3, keepdims=True) 


# In[9]:


X_test_gray.shape


# In[10]:


X_train_gray.shape


# In[11]:


X_validation_gray.shape


# In[12]:


X_train_gray_norm = (X_train_gray - 128)/128 
X_test_gray_norm = (X_test_gray - 128)/128
X_validation_gray_norm = (X_validation_gray - 128)/128


# In[13]:


i=234
plt.imshow(X_train_gray[i].squeeze(),cmap='gray')
plt.figure()
plt.imshow(X_train[i])
plt.figure()
plt.imshow(X_train_gray_norm[i].squeeze())


# # Data model

# In[14]:


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


# In[15]:


cnn_model= Sequential()
cnn_model.add(Conv2D(filters=6,kernel_size=(5,5),activation='relu',input_shape=(32,32,1)))
cnn_model.add(AveragePooling2D())

cnn_model.add(Conv2D(filters=16,kernel_size=(5,5),activation='relu'))
cnn_model.add(AveragePooling2D())

cnn_model.add(Flatten())

cnn_model.add(Dense(units=120,activation='relu'))

cnn_model.add(Dense(units=84,activation='relu'))

cnn_model.add(Dense(units=43,activation='softmax'))


# In[16]:


cnn_model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])


# In[17]:


history = cnn_model.fit(X_train_gray_norm,
                        y_train,
                        batch_size=500,
                        epochs=50,
                        verbose=1,
                        validation_data = (X_validation_gray_norm,y_validation))


# # Model evaluation

# In[18]:


score = cnn_model.evaluate(X_test_gray_norm,y_test)
print('Test Accuracy: {}'.format(score[1]))


# In[19]:


history.history.keys()


# In[20]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[21]:


epochs = range(len(accuracy))
plt.plot(epochs,accuracy,'bo',label='Training Accuracy')
plt.plot(epochs,val_accuracy,'b',label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()


# In[22]:


plt.plot(epochs,loss,'ro',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.title('Training and Validation Accuracy')
plt.legend()


# In[23]:


predict_classes=cnn_model.predict_classes(X_test_gray_norm)
y_true=y_test


# # Confusion matrix

# In[24]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,predict_classes)
plt.figure(figsize=(25,25))
sns.heatmap(cm,annot=True)


# In[25]:


L=7
W=7
fig,axes = plt.subplots(L,W,figsize=(12,12))
axes = axes.ravel()
for i in np.arange(0,L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(predict_classes[i],y_true[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=1)    


# # Model saving

# In[26]:


import os 
directory = os.path.join(os.getcwd(),'Model')

if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory,'Weight files.h5')
cnn_model.save(model_path)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




