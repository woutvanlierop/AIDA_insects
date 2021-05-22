#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from utils import get_label_from_image_path, load_image_array, check_str, resize_image, get_labelencoder_mapping
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import VGG16, DenseNet121
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, Conv2D,MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

datadir = Path("./sorted/")

nb_classes = 8


# In[21]:


input_dim = 150 # this is for square images, define two different input dimensions otherwise and change input_shape below

base_model = DenseNet121(include_top=False, weights='imagenet', 
                        input_shape=(input_dim,input_dim,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)



# In[22]:


#model.summary()


# In[23]:


from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
loss = CategoricalCrossentropy(
    name='categorical_crossentropy'
)

from tensorflow.keras.optimizers import Adam

# optimizer = Adam(learning_rate=0.0001)


# In[24]:


# optimizer = keras.optimizers.Adam(lr=0.0001, clipnorm = 1)
model.compile(loss=loss,
              optimizer='nadam',
              metrics=['Accuracy'])


# In[25]:


train_datagen = ImageDataGenerator(
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True, 
                                rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'sorted/train',
        target_size=(150, 150),
        batch_size=16,
        class_mode='categorical',
        shuffle=True)
validation_generator = test_datagen.flow_from_directory(
        'sorted/val',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)


callbacks_list = [ModelCheckpoint(monitor = 'val_loss',
                                  filepath = 'cnn_basic.h5',
                                  save_best_only = True,
                                  save_weights_only = False,
                                  verbose = 1),
                  EarlyStopping(monitor = 'val_loss',
                                patience = 9,
                                verbose = 1),
                  ReduceLROnPlateau(monitor = 'val_loss',
                                    factor = 0.1,
                                    patience = 4,
                                    verbose = 1)]

# In[ ]:


model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,
        validation_data=validation_generator,
        callbacks = callbacks_list,
        validation_steps=len(validation_generator))


# In[ ]:


validation_generator.reset()
pred = model.predict(validation_generator, verbose=1)

y_pred = np.argmax(pred,1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_true = le.fit_transform(pd.Series(validation_generator.filenames).apply(lambda x: x.split('/')[0]))

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import seaborn as sns
labelticks =['bl','c','k','m','sw','t','v','wmv']
plt.figure(figsize=(16,8))
sns.heatmap(confusion_matrix(y_true=y_true, y_pred=y_pred, normalize = 'true'), annot=True, xticklabels=labelticks, yticklabels=labelticks, fmt='.0%');
plt.title(f"Accuracy score: {balanced_accuracy_score(y_true=y_true, y_pred=y_pred)*100.:.0f}%");
plt.savefig('confusion_matrix_8.png')

print('balanced acc')
print(balanced_accuracy_score(y_true=y_true, y_pred=y_pred))

print('normal acc')
print(accuracy_score(y_true=y_true, y_pred=y_pred))