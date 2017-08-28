import pandas as pd
import json
import os
from pathlib import Path

from sklearn.model_selection import train_test_split

from keras.models import Sequential, model_from_json
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPool2D
from keras.optimizers import Adam

from load_data import batch_generator, input_shape
from load_data import fileDataPath, fileDataCSV


fileModelJSON = 'model.json'
fileWeights = 'model.h5'

# Training parameters
batch_size = 32
epochs = 5
learning_rate = 1e-4


# Load training data and split it into training and validation set
data = pd.read_csv(os.path.join(fileDataPath, fileDataCSV))
X_data = data[['center', 'left', 'right']].values
y_data = data['steering'].values

X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.2, random_state=10)


# Compile and train the model using the generator function
train_generator = batch_generator(X_train, y_train, batch_size=batch_size)
validation_generator = batch_generator(X_valid, y_valid, batch_size=batch_size)


# Load previous session model and retrain if model.json and model.h5 exists
if Path(fileModelJSON).is_file():
    with open(fileModelJSON) as jfile:
        model = model_from_json(json.load(jfile))

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    model.load_weights(fileWeights)
    print("Load model from disk:")
    model.summary()

# re-create model and restart training
else:

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape))
    model.add(Convolution2D(24, (5,5), strides=(2,2), activation="relu"))
    model.add(Convolution2D(36, (5,5), strides=(2,2), activation="relu"))
    model.add(Convolution2D(48, (5,5), strides=(2,2), activation="relu"))
    model.add(Convolution2D(64, (3,3), strides=(1,1), activation="relu"))
    model.add(Convolution2D(64, (3,3), strides=(1,1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    print("Create model:")
    model.summary()

# Save model and weights to disk
history = model.fit_generator(train_generator, samples_per_epoch=len(X_train),\
          epochs=epochs, validation_data=validation_generator, validation_steps=len(X_valid), verbose=1)

print("Save model to disk:")
if Path(fileModelJSON).is_file():
    os.remove(fileModelJSON)
json_string = model.to_json()
with open(fileModelJSON, 'w') as jfile:
    json.dump(json_string, jfile)

if Path(fileWeights).is_file():
    os.remove(fileWeights)
model.save_weights(fileWeights)

