# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 16:03:12 2020

The construction and execution of a neural network to classify the occurence of
magnetotail reconnection events in Saturn's tail within the Cassini MAG
observations.

This script reads in Cassini data and an events catalogue of plasmoids and TCRs,
it then creates a training, test, and validation for inputs into a keras NN model.

Goal: Correctly Identify and classify magnetic reconnection events in Cassini data

NOTE: This requires the installation of the tensorflow and keras packages, which
can be a pain sometimes.

@author: tmg1v19
"""

# import libraries 
from datetime import datetime
import keras
from keras.layers import Dense, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split

#=====FUNCTIONS=====#
# Set up output layer into a 2 variable vector, one variable per digit
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 2))

    for i in range(len(y)):
        y_vect[i, y[i]] = 1.0

    return y_vect

#=====Classes=====#
# A class to store data on the progression of NN acuracy
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

#=====INITIALIZATION=====#
# how far backward and forward (in minutes) should be included as ML inputs
# (Currently recommend keeping t_pre_event + t_post_event ≈ 60)
t_pre_event = 60
t_post_event = 0

# establish of NN execution parameters
batch_size = 128
num_classes = 2
epochs = 10

# initialize an empty list for the ML input data
Cass_ML_input = []

# identify directory location of THIS file
dir_path = os.path.dirname(os.path.realpath(__file__))

#=====DATA READING AND PROCESSING=====#
# read in the Cassini MAG data into a pandas data array
Cass_data = pd.read_csv(dir_path + "/Cassini_Observations/Cass_data.txt", 
                        sep = "\t", header = 4947)

# convert pandas indexing from iteration based to date based and interpolating
# gaps in data
Cass_data["Timestamp(UTC)"] = pd.to_datetime(Cass_data["Timestamp(UTC)"], 
         format="%d/%m/%Y %H:%M:%S.%f")
Cass_data = Cass_data.set_index("Timestamp(UTC)")
Cass_data = Cass_data.resample("1T",loffset="30s").mean().interpolate()

# read in a catalogue of magnetotail events
smith_catalogue = pd.read_csv(r"C:\Users\tmg1v19\Post-doc\Smith2016ReconnectionCatalogue.csv")

# set up an empty array for the classification of events
Events = np.empty(len(Cass_data.index))

# loop through each event in the catalogue
for i in range(len(smith_catalogue["EventDateTime"])):
    
    # Fill events arrays with corresponding boolean representation
    Events[np.where(
            (Cass_data.index > datetime.strptime(smith_catalogue["EventStartDateTime"][i],
                                                 "%d/%m/%Y %H:%M")) & 
            (Cass_data.index < datetime.strptime(smith_catalogue["EventEndDateTime"][i],
                                                 "%d/%m/%Y %H:%M"))
            )] = 1.0

# store the indices for every event from the input catalogue
event_index = np.where(Events == 1.0)

# Populate ML input array with Cassini Magnetic data corresponding to events
# occuring
for i in event_index[0]:
    Cass_ML_input.append([ 
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BX_KRTP(nT)"],
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BY_KRTP(nT)"],
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BZ_KRTP(nT)"]
            ])

bool_1s = np.shape(Cass_ML_input)[0]

# initialize a count for number of no events found
cnt_0s=0

x=[]

# Populate ML input array with Cassini Magnetic data corresponding to no events
# occuring
while (bool_1s+cnt_0s) < (2*bool_1s):
    
    n = random.randrange(0 + t_pre_event, len(Events) - t_post_event)
    if np.max(Events[n - t_pre_event : n + t_post_event]) < 0.5:
        x.append(np.mean(Events[n - t_pre_event : n + t_post_event]))
        Cass_ML_input.append([
                Cass_data.iloc[n - t_pre_event : n + t_post_event]["BX_KRTP(nT)"],
                Cass_data.iloc[n - t_pre_event : n + t_post_event]["BY_KRTP(nT)"],
                Cass_data.iloc[n - t_pre_event : n + t_post_event]["BZ_KRTP(nT)"]
                ])
        cnt_0s+=1

# Convert ML input from a python list to a numpy array
Cass_ML_input = np.array(Cass_ML_input)

# Create a corresponding numpy boolean array describing event type 
# (1 = event, 0 = no event) 
Event_ML_scalers = np.concatenate((np.ones((bool_1s,), dtype = int), 
                                   np.zeros((cnt_0s,), dtype = int)))

# Convert boolean scaler array into boolean vector/tensor array
Event_ML_outputs = convert_y_to_vect(Event_ML_scalers)

Input_test_train, Input_val, Outp_test_train, Outp_val = train_test_split(Cass_ML_input, 
                                                                          Event_ML_outputs, 
                                                                          test_size=0.2)

#=============================#
#=====Keras CNN Structure=====#
#=============================#

# State model will be built in sequential order
model = Sequential()

# Flatten to a 1D tensor
model.add(Flatten())

# Create a hidden layer of size N=1000, calculations use the "relu" activation
# function
model.add(Dense(50, activation='relu'))

# Create an output layer  of size N=num_classes, calculations use the "softmax"
# activation function
model.add(Dense(num_classes, activation='softmax'))

# Now that the model construction is finished, compile the model to glue all the
# layers together with loss calculated using cross entropy, and the optimizer 
# being an ADAM descent model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# NOTE: No weights or biases have been established in this construction,
#       keras does that for us.

#=================================#
#=====Keras CNN Structure End=====#
#=================================#

# Store the values of accuracy at each epoch to the "history" class 
history = AccuracyHistory()

# Train our model  with the given initial parameters with the intent to optimize
# the model to fit both the training and test sets
hist = model.fit(Input_test_train, Outp_test_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=(1./4.),
          callbacks=[history])

# Check how well the model performs on the test set after it's construction.
# For a better constructed NN this score should reflect a seperate validation
# set
score = model.evaluate(Input_val, Outp_val, verbose=1)

# Output the loss of the model's weights and biases and the accuracy of the model
# on the "score" dataset
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#=====PLOTTING PROCEDURES=====#

# Summarize history for accuracy
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# summarize history for loss
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


"""
gridsize=(2,1)

fig=plt.figure(figsize=(20,20))

ax1 = plt.subplot2grid(gridsize,(0,0))
plt.title("2006 Cassini Magnetic Measurements")
ax2 = plt.subplot2grid(gridsize,(1,0))

ax1.plot(Cass_data["BY_KRTP(nT)"])
ax1.set_ylabel("$|B|\ (nT)$")
ax1.set_xlabel("")
ax1.plot(Cass_data.index, Events)

ax2.plot(Cass_data.index, Events)
ax2.set_ylabel("Event")
ax2.set_xlabel("Date")
plt.show()

"""

#=============#
#=====EOF=====#
#=============#