###########################################################################
## Copyright 2021 Hewlett Packard Enterprise Development LP
## Licensed under the Apache License, Version 2.0 (the "License"); you may
## not use this file except in compliance with the License. You may obtain
## a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
############################################################################

import tensorflow as tf
import numpy as np
import time
import datetime
from swarm import SwarmCallback
import os
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import * 
from keras.preprocessing import image
default_max_epochs = 5
default_min_peers = 2

def main():
  dataDir = os.getenv('DATA_DIR', './data')
  modelDir = os.getenv('MODEL_DIR', './model')
  modelDir = os.getenv('MODEL_DIR', './model')
  max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
  min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))

  model_name = 'covid_tf'
  
  #Defining paths
  TRAIN_PATH = "CovidDataset/CovidDataset/Train"
  VAL_PATH = "CovidDataset/CovidDataset/Val"
  
  #Moulding train images
  train_datagen = image.ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True)
  test_dataset = image.ImageDataGenerator(rescale=1./255)
  
  #Reshaping test and validation images 
  train_generator = train_datagen.flow_from_directory(
      TRAIN_PATH,
      target_size = (224,224),
      batch_size = 32,
      class_mode = 'binary')
  validation_generator = test_dataset.flow_from_directory(
      VAL_PATH,
      target_size = (224,224),
      batch_size = 32,
      class_mode = 'binary')

  #creating CNN model
  model = Sequential()
  model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
  model.add(Conv2D(128,(3,3),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64,(3,3),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128,(3,3),activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(64,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1,activation='sigmoid'))

  model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])
 

  # Create Swarm callback
  swarmCallback = SwarmCallback(sync_interval=128,
                                min_peers=min_peers,
                                val_data=validation_generator,
                                val_batch_size=32,
                                model_name=model_name)

  model.fit train_generator, 
            batch_size = 64,
            epochs=max_epochs,
            verbose=1,            
            callbacks=[swarmCallback])
  #Getting summary
  summary=hist_new.history
  print(summary)
  model.save("model_covid.h5")
  # Save model and weights
  model_path = os.path.join(modelDir, model_name)
  model.save(model_path)
  print('Saved the trained model!')

if __name__ == '__main__':
  main()
