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

Current best structure:
-----------------------
    t_pre_event = 15 
    t_post_event = 15
    batch_size = 128
    num_classes = 2
    epochs = 10
    hidden_layer_nodes_1 = 50
    hidden_layer_nodes_2 = 25
    catalogue_pruning = False
    normalization = False
    
    Accuracy = ~0.93
    Loss = ~0.19

To Do:
------

- Exclude small reconnections
- Optimize pre/post times - Current +/-15 mins

Feed in 2006/09/10 only for non-events
Magnetopause crossings
Titan flybys

exclude(15Rs, Dayside)
Histogram of sampling with radius,lat,long for all, cat dates, comparing event times

SVM? - code created
RFR?
Decision Tree?

@author: tmg1v19
"""

# import libraries 
from datetime import datetime
import keras
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

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
t_pre_event = 15
t_post_event = 15

# establish NN execution parameters
batch_size = 128
num_classes = 2
epochs = 100
hidden_layer_nodes_1 = 40
hidden_layer_nodes_2 = 20
hidden_layer_nodes_3 = 10

random.seed(42)

# initialize an empty list for the ML input data
Cass_ML_train , Cass_ML_test, Cass_ML_val= [], [], []

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




crossings = pd.read_csv(dir_path + "/MP_BS_Crossings.txt")

location = np.zeros(Cass_data.shape[0])+2
SCAS = 0
C_dats=Cass_data.index

for i in range(len(crossings)):
    YR = str(crossings['Year'][i])
    DOY = str(crossings['DOY'][i])
    HR = str(crossings['Hour'][i])
    MIN = str(crossings['Minute'][i])
    
    date = datetime.strptime(YR+' '+DOY+' '+HR+' '+MIN, '%Y %j %H %M')
    
    if crossings['TYPE_DIRECTION'][i] == 'BSI':
        location[np.where(C_dats == str(date.year)+'-'+str(date.month)+'-'+str(date.day)+' '+str(date.hour)+':'+str(date.minute)+':30')[0][0]:]=1
    elif crossings['TYPE_DIRECTION'][i] == 'BSO':
        location[np.where(C_dats == str(date.year)+'-'+str(date.month)+'-'+str(date.day)+' '+str(date.hour)+':'+str(date.minute)+':30')[0][0]:]=2
    elif crossings['TYPE_DIRECTION'][i] == 'MPI':
        location[np.where(C_dats == str(date.year)+'-'+str(date.month)+'-'+str(date.day)+' '+str(date.hour)+':'+str(date.minute)+':30')[0][0]:]=0
    elif crossings['TYPE_DIRECTION'][i] == 'MPO':
        location[np.where(C_dats == str(date.year)+'-'+str(date.month)+'-'+str(date.day)+' '+str(date.hour)+':'+str(date.minute)+':30')[0][0]:]=1
    else:
        print(SCAS)
        SCAS+=1


# read in Cass data in KSM units
Cass_data_ksm = pd.read_csv(dir_path + "/Cassini_Observations/Cass_data_ksm.txt", 
                        sep = "\t", header = 5008)

Cass_data_ksm = Cass_data_ksm.drop(np.arange(83386))

# convert pandas indexing from iteration based to date based and interpolating
# gaps in data
Cass_data_ksm["Timestamp(UTC)"] = pd.to_datetime(Cass_data_ksm["Timestamp(UTC)"], 
         format="%d/%m/%Y %H:%M:%S.%f")
Cass_data_ksm = Cass_data_ksm.set_index("Timestamp(UTC)")
Cass_data_ksm = Cass_data_ksm.resample("1T",loffset="30s").mean().interpolate()

x = Cass_data_ksm["X_KSM(km)"]
y = Cass_data_ksm["Y_KSM(km)"]
z = Cass_data_ksm["Z_KSM(km)"]
rs = 58232

r = np.sqrt(x**2 + y**2 + z**2)/rs
t = np.degrees(np.arccos(z/(rs*r)))
p = np.degrees(np.arctan2(y, x))+180.

lt=p*24./360 %24

# read in a catalogue of magnetotail events
smith_catalogue = pd.read_csv(dir_path + "/Smith2016ReconnectionCatalogue.csv")

# set up an empty array for the classification of events
Events = np.empty(len(Cass_data.index))


# loop through each event in the catalogue
for i in range(len(smith_catalogue["EventDateTime"])):
    rand = random.random()
    # Fill events arrays with seperated train/test/val representation
    if rand < 0.6:
        Events[np.where(
                    (Cass_data.index > datetime.strptime(smith_catalogue["EventStartDateTime"][i],
                                                     "%d/%m/%Y %H:%M")) & 
                    (Cass_data.index < datetime.strptime(smith_catalogue["EventEndDateTime"][i],
                                                     "%d/%m/%Y %H:%M"))
                    )] = 1.0
    elif 0.6 <= rand < 0.8:
        Events[np.where(
                    (Cass_data.index > datetime.strptime(smith_catalogue["EventStartDateTime"][i],
                                                     "%d/%m/%Y %H:%M")) & 
                    (Cass_data.index < datetime.strptime(smith_catalogue["EventEndDateTime"][i],
                                                     "%d/%m/%Y %H:%M"))
                    )] = 2.0
    else:
        Events[np.where(
                    (Cass_data.index > datetime.strptime(smith_catalogue["EventStartDateTime"][i],
                                                     "%d/%m/%Y %H:%M")) & 
                    (Cass_data.index < datetime.strptime(smith_catalogue["EventEndDateTime"][i],
                                                     "%d/%m/%Y %H:%M"))
                    )] = 3.0


unsearched_events = np.copy(Events)
# store the indices for every event from the input catalogue
train_event_index = np.where(Events == 1.0)
test_event_index = np.where(Events == 2.0)
val_event_index = np.where(Events == 3.0)

Events[np.where(r < 15)] = np.nan
Events[np.where(np.logical_and(lt < 18, lt > 6))] = np.nan

# establish the temporal limitations of the Smith catalogue
catalogue_lims = [datetime(2006,1,1), datetime(2007,1,1), datetime(2009,10,7), datetime(2011,1,1)]

# Data normalization
"""
r_gt_15 = np.where(r > 15)

Cass_data["BX_KRTP(nT)"] =( (Cass_data["BX_KRTP(nT)"] - 
         np.min(Cass_data.iloc[r_gt_15]["BX_KRTP(nT)"])) / 
            (np.max(Cass_data.iloc[r_gt_15]["BX_KRTP(nT)"]) - 
             np.min(Cass_data.iloc[r_gt_15]["BX_KRTP(nT)"]))
            )
            
Cass_data["BY_KRTP(nT)"] =( (Cass_data["BY_KRTP(nT)"] - 
         np.min(Cass_data.iloc[r_gt_15]["BY_KRTP(nT)"])) / 
            (np.max(Cass_data.iloc[r_gt_15]["BY_KRTP(nT)"]) - 
             np.min(Cass_data.iloc[r_gt_15]["BY_KRTP(nT)"]))
            )

Cass_data["BZ_KRTP(nT)"] =( (Cass_data["BZ_KRTP(nT)"] - 
         np.min(Cass_data.iloc[r_gt_15]["BZ_KRTP(nT)"])) / 
            (np.max(Cass_data.iloc[r_gt_15]["BZ_KRTP(nT)"]) - 
             np.min(Cass_data.iloc[r_gt_15]["BZ_KRTP(nT)"]))
            )
"""
print("Seperating training and validation sets")

# Populate ML input array with Cassini Magnetic data corresponding to events
# occuring
for i in train_event_index[0]:
    Cass_ML_train.append([
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BX_KRTP(nT)"],
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BY_KRTP(nT)"],
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BZ_KRTP(nT)"]
            ])
for i in test_event_index[0]:
    Cass_ML_test.append([
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BX_KRTP(nT)"],
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BY_KRTP(nT)"],
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BZ_KRTP(nT)"]
            ])
for i in val_event_index[0]:
    Cass_ML_val.append([ 
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BX_KRTP(nT)"],
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BY_KRTP(nT)"],
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BZ_KRTP(nT)"]
            ])

# identify the number of samples in the train/test/validation sets
bool_1s_train = np.shape(Cass_ML_train)[0]
bool_1s_test = np.shape(Cass_ML_test)[0]
bool_1s_val = np.shape(Cass_ML_val)[0]

# initialize a count for number of null events found
cnt_0s_train=0
cnt_0s_test=0
cnt_0s_val=0

print("Populating zeroes, {}, {}, {}".format(bool_1s_train, bool_1s_test, bool_1s_val))
iters = 0

# Populate ML input array with Cassini Magnetic data within the Smith Catalogue timerange
# corresponding to null events occuring

ran_range_low= np.where(Cass_data.index > catalogue_lims[0])[0][0]
ran_range_hi= np.where(Cass_data.index > catalogue_lims[-1])[0][0]


while (bool_1s_train + cnt_0s_train) < (1.25*bool_1s_train):
    n = random.randrange(0, train_event_index[0].shape[0])
    n = train_event_index[0][n]-random.randrange(1,10)
    
    if (Cass_data.index[n] < catalogue_lims[1] or Cass_data.index[n] > catalogue_lims[2]):
        if (Events[n] == 0.0 and location[n] == 0):
            Cass_ML_train.append([
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BX_KRTP(nT)"],
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BY_KRTP(nT)"],
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BZ_KRTP(nT)"]
                    ])
            Events[n] = -1.0
            cnt_0s_train += 1

    iters += 1
    if iters % 10000 == 0:
        print("Train Iterations near Event = {}".format(iters))
        print("Counts = {}".format(cnt_0s_train))
        
while (bool_1s_train + cnt_0s_train) < (1.5*bool_1s_train):
    n = random.randrange(0, train_event_index[0].shape[0])
    n = train_event_index[0][n]+random.randrange(1,10)
    
    if (Cass_data.index[n] < catalogue_lims[1] or Cass_data.index[n] > catalogue_lims[2]):
        if (Events[n] == 0.0 and location[n] == 0):
            Cass_ML_train.append([
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BX_KRTP(nT)"],
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BY_KRTP(nT)"],
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BZ_KRTP(nT)"]
                    ])
            Events[n] = -1.0
            cnt_0s_train += 1

    iters += 1
    if iters % 10000 == 0:
        print("Train Iterations near Event = {}".format(iters))
        print("Counts = {}".format(cnt_0s_train))



while (bool_1s_train + cnt_0s_train) < (2*bool_1s_train):
    n = random.randrange(ran_range_low, ran_range_hi)
    
    if (Cass_data.index[n] < catalogue_lims[1] or Cass_data.index[n] > catalogue_lims[2]):
        if (Events[n] == 0.0 and location[n] == 0):
            Cass_ML_train.append([
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BX_KRTP(nT)"],
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BY_KRTP(nT)"],
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BZ_KRTP(nT)"]
                    ])
            Events[n] = -1.0
            cnt_0s_train += 1

    iters += 1
    if iters % 10000 == 0:
        print("Train Iterations MSph = {}".format(iters))
        print("Counts = {}".format(cnt_0s_train))



while (bool_1s_train + cnt_0s_train) < (3*bool_1s_train):
    n = random.randrange(0, len(Cass_data))
    
    if (Events[n] == 0.0 and location[n] == 1):
        Cass_ML_train.append([
                Cass_data.iloc[n - t_pre_event : n + t_post_event]["BX_KRTP(nT)"],
                Cass_data.iloc[n - t_pre_event : n + t_post_event]["BY_KRTP(nT)"],
                Cass_data.iloc[n - t_pre_event : n + t_post_event]["BZ_KRTP(nT)"]
                ])
        Events[n] = -1.0
        cnt_0s_train += 1

    iters += 1
    if iters % 10000 == 0:
        print("Train Iterations MSh = {}".format(iters))
        print("Counts = {}".format(cnt_0s_train))
        
while (bool_1s_train + cnt_0s_train) < (4*bool_1s_train):
    n = random.randrange(0, len(Cass_data))
    
    if (Events[n] == 0.0 and location[n] == 2):
        Cass_ML_train.append([
                Cass_data.iloc[n - t_pre_event : n + t_post_event]["BX_KRTP(nT)"],
                Cass_data.iloc[n - t_pre_event : n + t_post_event]["BY_KRTP(nT)"],
                Cass_data.iloc[n - t_pre_event : n + t_post_event]["BZ_KRTP(nT)"]
                ])
        Events[n] = -1.0
        cnt_0s_train += 1

    iters += 1
    if iters % 10000 == 0:
        print("Train Iterations SW = {}".format(iters))
        print("Counts = {}".format(cnt_0s_train))



while (bool_1s_val + cnt_0s_val) < (2*bool_1s_val):
    n = random.randrange(ran_range_low, ran_range_hi)

    if (Cass_data.index[n] < catalogue_lims[1] or Cass_data.index[n] > catalogue_lims[2]):
        if Events[n] == 0.0:
            Cass_ML_val.append([
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BX_KRTP(nT)"],
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BY_KRTP(nT)"],
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BZ_KRTP(nT)"]
                    ])
            Events[n] = -1.0
            cnt_0s_val += 1

    iters += 1    
    if iters % 10000 == 0:
        print("Val Iterations = {}".format(iters))
        print("Counts = {}".format(cnt_0s_val))

while (bool_1s_test + cnt_0s_test) < (2*bool_1s_test):
    n = random.randrange(ran_range_low, ran_range_hi)

    if (Cass_data.index[n] < catalogue_lims[1] or Cass_data.index[n] > catalogue_lims[2]):
        if Events[n] == 0.0:
            Cass_ML_test.append([
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BX_KRTP(nT)"],
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BY_KRTP(nT)"],
                    Cass_data.iloc[n - t_pre_event : n + t_post_event]["BZ_KRTP(nT)"]
                    ])
            Events[n] = -1.0
            cnt_0s_test += 1

    iters += 1    
    if iters % 10000 == 0:
        print("Test Iterations = {}".format(iters))
        print("Counts = {}".format(cnt_0s_test))

# Convert ML input from a python list to a numpy array
Cass_ML_train = np.array(Cass_ML_train)
Cass_ML_test = np.array(Cass_ML_test)
Cass_ML_val = np.array(Cass_ML_val)

# Create a corresponding numpy boolean array describing event type 
# (1 = event, 0 = no event) 
Event_ML_scalers_t = np.concatenate((np.ones((bool_1s_train,), dtype = int), 
                                   np.zeros((cnt_0s_train,), dtype = int)))
Event_ML_scalers_te = np.concatenate((np.ones((bool_1s_test,), dtype = int), 
                                   np.zeros((cnt_0s_test,), dtype = int)))
Event_ML_scalers_v = np.concatenate((np.ones((bool_1s_val,), dtype = int), 
                                   np.zeros((cnt_0s_val,), dtype = int)))

# Convert boolean scaler array into boolean vector/tensor array
Event_ML_train = convert_y_to_vect(Event_ML_scalers_t)
Event_ML_test = convert_y_to_vect(Event_ML_scalers_te)
Event_ML_val = convert_y_to_vect(Event_ML_scalers_v)


# Shuffle train/test/val sets
shuf = list(range(len(Event_ML_train)))
shuf_ind=shuffle(shuf)

Event_ML_train, Cass_ML_train = Event_ML_train[shuf_ind,:], Cass_ML_train[shuf_ind,:,:]

shuf = list(range(len(Event_ML_test)))
shuf_ind=shuffle(shuf)

Event_ML_test, Cass_ML_test = Event_ML_test[shuf_ind,:], Cass_ML_test[shuf_ind,:,:]

shuf = list(range(len(Event_ML_val)))
shuf_ind=shuffle(shuf)

Event_ML_val, Cass_ML_val = Event_ML_val[shuf_ind,:], Cass_ML_val[shuf_ind,:,:]


print("Training, test and validation sets finished.")

#=============================#
#=====Keras NN Structure=====#
#=============================#

# State model will be built in sequential order
model = Sequential()

# Flatten to a 1D tensor
model.add(Flatten())

# Create a hidden layer of size dependant on an input variable, calculations 
# use the "relu" activation function
model.add(Dense(hidden_layer_nodes_1, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(hidden_layer_nodes_2, activation='relu'))
model.add(Dropout(0.3))
#model.add(Dense(hidden_layer_nodes_3, activation='relu'))

# Create an output layer  of size N=num_classes, calculations use the "softmax"
# activation function
model.add(Dense(num_classes, activation='softmax'))

# Now that the model construction is finished, compile the model to glue all the
# layers together with loss calculated using cross entropy, and the optimizer 
# being an ADAM descent model
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer = keras.optimizers.Adam(),
              metrics = ['binary_accuracy'])

# NOTE: No weights or biases have been established in this construction,
#       keras does that for us.

#=================================#
#=====Keras CNN Structure End=====#
#=================================#

# Store the values of accuracy at each epoch to the "history" class 
history = AccuracyHistory()


# Split the data
#Cass_ML_in, Cass_ML_test, Event_ML_in, Event_ML_test = train_test_split(Cass_ML_train, Event_ML_train, test_size=0.25, shuffle= True)

# Train our model  with the given initial parameters with the intent to optimize
# the model to fit both the training and test sets
hist = model.fit(Cass_ML_train, Event_ML_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(Cass_ML_test, Event_ML_test),
          callbacks=[history])

# Check how well the model performs on the test set after it's construction.
# For a better constructed NN this score should reflect a seperate validation
# set
score = model.evaluate(Cass_ML_val, Event_ML_val, verbose=1)

# Output the loss of the model's weights and biases and the accuracy of the model
# on the "score" dataset
print('NN Test loss:', score[0])
print('NN Test accuracy:', score[1])


#=====PLOTTING PROCEDURES=====#

# Summarize history for accuracy
plt.figure()
plt.plot(hist.history['binary_accuracy'])
plt.plot(hist.history['val_binary_accuracy'])
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

#=====PREDICTIONS AND CONFUSION MATRICES=====#

# Training
classification = model.predict(Cass_ML_train, batch_size = 128, verbose = 1)

x = np.where(classification[:,1] > classification[:,0])
pred = np.zeros(classification.shape[0])
pred[x] = 1

x = np.where(Event_ML_train[:,1] > Event_ML_train[:,0])
obs = np.zeros(classification.shape[0])
obs[x] = 1
train_conf = confusion_matrix(obs, pred)

# Test
classification = model.predict(Cass_ML_test, batch_size = 128, verbose = 1)

x = np.where(classification[:,1] > classification[:,0])
pred = np.zeros(classification.shape[0])
pred[x] = 1

x = np.where(Event_ML_test[:,1] > Event_ML_test[:,0])
obs = np.zeros(classification.shape[0])
obs[x] = 1
test_conf = confusion_matrix(obs, pred)

# Validation
classification = model.predict(Cass_ML_val, batch_size = 128, verbose = 1)

x = np.where(classification[:,1] > classification[:,0])
pred = np.zeros(classification.shape[0])
pred[x] = 1

x = np.where(Event_ML_val[:,1] > Event_ML_val[:,0])
obs = np.zeros(classification.shape[0])
obs[x] = 1
val_conf = confusion_matrix(obs, pred)


# Create an input for the entirety of 2010

Cass_predict=[]

cat_start = np.array([15,2374771,4749541]) #np.array([3071519])
cat_end = np.array([2374770,4749540,7124296]) #np.array([3597118])
ml_ev=np.array([])

for n in range(len(cat_start)):
    fy_start = cat_start[n]
    fy_end = cat_end[n]
    
    for i in range(fy_start,fy_end):
        Cass_predict.append([ 
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BX_KRTP(nT)"],
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BY_KRTP(nT)"],
            Cass_data.iloc[i - t_pre_event : i + t_post_event]["BZ_KRTP(nT)"]
            ])
        if i % 100000 == 0:
            print(i)
    
    Cass_predict=np.array(Cass_predict)

    print(Cass_predict.shape)


    classification = model.predict(Cass_predict, batch_size = 128, verbose = 1)

    ml_events = np.argmax(classification,axis = 1)
    ml_ev = np.concatenate((ml_ev,ml_events))
    del Cass_predict
    Cass_predict=[]

pred = ml_events
unsearched_events[np.where(np.isnan(unsearched_events))]=0
obs = unsearched_events[fy_start:fy_end]
obs[np.where(obs > 0.5)[0]] = 1.0
obs[np.where(obs < -0.5)[0]] = 0.0
full_year_conf = confusion_matrix(obs, pred)

TP = full_year_conf[1][1]
TN = full_year_conf[0][0]
FP = full_year_conf[0][1]
FN = full_year_conf[1][0]

HSS = 2.*((TP*TN) - (FN*FP))/((TP + FN)*(FN + TN) + (TP + FP)*(FP + TN)) # Heidke Skill Score
TSS = (TP/(TP + FN)) - (FP/(FP + TN)) # True Skill Score
BIAS = (TP + FP)/(TP + FN) # Bias of Detection
POD = TP/(TP + FN) # Probability of Detection
POFD = FP/(TN + FP) # Probablity of False Detection
FAR = FP/(TP + FP) # False Alarm Ratio
TS = TP/(TP + FN + FP) # Threat Score
OR = (TP*TN)/(FN*FP) # Odds Ratio

starts_ev = np.where(ml_ev[:-1] < ml_ev[1:])[0] + 1 + cat_start[0]
ends_ev = np.where(ml_ev[:-1] > ml_ev[1:])[0] + 1 +cat_start[0]

starts_date_ev = Cass_data.index[starts_ev]
ends_date_ev = Cass_data.index[ends_ev]

dbth_ev = np.array([])
direction_ev = np.array([])
for i in range(len(starts_ev)):
    ev_max = np.max(Cass_data["BY_KRTP(nT)"].iloc[starts_ev[i]:ends_ev[i] + 1])
    ev_min = np.min(Cass_data["BY_KRTP(nT)"].iloc[starts_ev[i]:ends_ev[i] + 1])
    loc_max = np.where(Cass_data["BY_KRTP(nT)"].iloc[starts_ev[i]:ends_ev[i] + 1] == ev_max)[0][0]
    loc_min = np.where(Cass_data["BY_KRTP(nT)"].iloc[starts_ev[i]:ends_ev[i] + 1] == ev_min)[0][0]
    
    direction = loc_min - loc_max
    
    if direction > 0:
        direction = 1
        direction_ev = np.append(direction_ev, "Tailward")
    else:
        direction = -1
        direction_ev = np.append(direction_ev, "Planetward")
    
    dbth_ev = np.append(dbth_ev,  direction*(ev_max - ev_min))

radius_ev = np.array(r[starts_ev])
lt_ev = np.array(lt[starts_ev])
theta_ev = np.array(t[starts_ev])
duration_ev = ends_ev - starts_ev

center_ev = ((starts_ev+ends_ev)/2).astype(int)
bth_rms=np.array([])
for i in center_ev:
    bth_rms = np.append(bth_rms, np.sqrt(np.mean(Cass_data["BY_KRTP(nT)"][i - 30 : i + 30]**2)))

rms_lim_ev = np.abs(dbth_ev)/bth_rms

df = pd.DataFrame({'Year' : starts_date_ev.year, 'DayOfYear' : starts_date_ev.dayofyear,'EventStartDateTime' : starts_date_ev, 'EventEndDateTime' : ends_date_ev, 'DeltaBTheta(nT)' : dbth_ev, 'EventDirection' : direction_ev, 'RadialDistance(RS)' : radius_ev, 'LocalTime(Hr)' : lt_ev, 'Latitude(deg)' : theta_ev, 'EventDuration(min)' : duration_ev,'dB_th/B_th_rms' : rms_lim_ev}, index = np.arange(len(starts_ev))+1)
df.to_csv('ML_2010_reconnection_catalogue.csv')

#======================#
#=====Supplimental=====#
#======================#


"""

x = Cass_data_ksm["X_KSM(km)"]
y = Cass_data_ksm["Y_KSM(km)"]
z = Cass_data_ksm["Z_KSM(km)"]
rs = 58232

e_2010=Events[3071518:(3071518+525599)]

event_index=np.where(Events[3071518:] == 1)

ml_events=np.argmax(classification,axis=1)
ml_event_index =np.where(ml_events == 1)

hits=e_2010*ml_events
fails=e_2010-ml_events

HI=np.where(hits == 1)
MI= np.where(fails == 1)
FDI= np.where(fails == -1)

r_fill=100
th_fill = np.arange(24)*2*np.pi/24

th_c = np.arange(100)*2*np.pi/100

x_fill=r_fill*np.cos(th_fill)
y_fill=r_fill*np.sin(th_fill)

plt.figure()

for i in range(0,len(x_fill),2):
    plt.fill([0,x_fill[i],x_fill[i+1]],[0,y_fill[i],y_fill[i+1]],color="grey", alpha = 0.1)

for i in range(10,100,10):
    rc = i
    
    xc = rc*np.cos(th_c)
    yc = rc*np.sin(th_c)
    
    plt.plot(xc,yc, color = 'k', linestyle = (0, (1, 10)), linewidth = 1)

plt.scatter(x[FDI[0]+3071518]/rs,y[FDI[0]+3071518]/rs,s=1, color = 'orange', 
            label = 'False Detections')
plt.scatter(x[HI[0]+3071518]/rs,y[HI[0]+3071518]/rs,s=1, color = 'g', label = 'Hits')
plt.scatter(x[MI[0]+3071518]/rs,y[MI[0]+3071518]/rs,s=1, color = 'r', label = 'Misses')
plt.scatter(0,0, color='k', s=100.0, marker = 'o', label = "Saturn")
plt.ylim([-20,70])
plt.xlim([-30,30])
plt.xlabel('X/R$_S$', fontsize = 20)
plt.ylabel('Y/R$_S$', fontsize = 20)
plt.title('Detections of ML algorithm', fontsize = 20)
plt.legend(fontsize = 16)


y1 = ml_events > 0.5
y1_shifted = np.roll(y1,1)
starts = y1 & ~y1_shifted
ends = ~y1 & y1_shifted
starts = np.nonzero(starts)[0]
ends = np.nonzero(ends)[0]

obs_b = obs > 0.5
obs_shifted = np.roll(obs_b,1)
smith_starts = obs_b & ~obs_shifted
smith_ends = ~obs_b & obs_shifted
smith_starts = np.nonzero(smith_starts)[0]
smith_ends = np.nonzero(smith_ends)[0]

dbth = np.array([])

if starts[0] > ends[0]:
    starts = starts[0:-1]
    ends = ends[1:]

s_2010 = starts + 3071519
e_2010 = ends + 3071519

s_s_2010 = smith_starts
s_e_2010 = smith_ends

smith_dbth = np.abs(smith_catalogue["DeltaBtheta"])
smith_events_dt = pd.to_datetime(smith_catalogue["EventDateTime"], format = "%d/%m/%Y %H:%M")
smith_dbth_2010 = np.abs(smith_catalogue["DeltaBtheta"].iloc[np.where(smith_events_dt > datetime(2010,1,1))[0]])

for i in range(len(starts)):
    dbth = np.append(dbth, np.max(Cass_data["BY_KRTP(nT)"].iloc[s_2010[i]:e_2010[i]]) - np.min(Cass_data["BY_KRTP(nT)"].iloc[s_2010[i]:e_2010[i]]))

plt.figure()

plt.hist(dbth, bins = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0])
plt.hist(smith_dbth, bins = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0])
plt.hist(smith_dbth_2010, bins = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0])

plt.xlabel("$B_\\theta\ (nT)$", fontsize = 20)
plt.ylabel('Counts', fontsize = 20)

plt.show()

TPS=np.array([])
dbth_tps = np.array([])
dbth_fps = np.array([])
dbth_tot = np.array([])
dbth_fns = np.array([])

for i in range(len(smith_starts)):
    if np.max(pred[s_s_2010[i]:s_e_2010[i]]) == 0:
        dbth_fns = np.append(dbth_fns, np.max(Cass_data["BY_KRTP(nT)"].iloc[s_2010[i]:e_2010[i]]) - np.min(Cass_data["BY_KRTP(nT)"].iloc[s_2010[i]:e_2010[i]]))

for i in range(len(starts)):
    TPS = np.append(TPS, np.max(Events[s_2010[i]:e_2010[i]]))
    
for i in range(len(starts)):
    dbth_tot = np.append(dbth_tot, np.max(Cass_data["BY_KRTP(nT)"].iloc[s_2010[i]:e_2010[i]]) - np.min(Cass_data["BY_KRTP(nT)"].iloc[s_2010[i]:e_2010[i]]))
    
    if TPS[i] == 1:
        dbth_tps = np.append(dbth_tps, np.max(Cass_data["BY_KRTP(nT)"].iloc[s_2010[i]:e_2010[i]]) - np.min(Cass_data["BY_KRTP(nT)"].iloc[s_2010[i]:e_2010[i]]))
    else:
        dbth_fps = np.append(dbth_fps, np.max(Cass_data["BY_KRTP(nT)"].iloc[s_2010[i]:e_2010[i]]) - np.min(Cass_data["BY_KRTP(nT)"].iloc[s_2010[i]:e_2010[i]]))

plt.figure()

plt.hist(dbth_fns, bins = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0], color = 'red', label = "FN")
plt.hist(dbth_tot, bins = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0], color = 'orange', label = "TP+FP")
plt.hist(dbth_tps, bins = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0], color = 'green', label = "TP")
plt.hist(dbth_fns, bins = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0], color = 'red')

plt.xlabel("$\Delta B_\\theta\ (nT)$", fontsize = 20)
plt.ylabel('Counts', fontsize = 20)
plt.title("Distributions of Latitudinal Magnetic Variation", fontsize = 20)
plt.yscale('log')
plt.legend(fontsize = 16)

plt.show()


big_fps = np.where((TPS == 0) & (dbth_tot >= 3.0))[0]

x = Cass_data_ksm["X_KSM(km)"]
y = Cass_data_ksm["Y_KSM(km)"]
z = Cass_data_ksm["Z_KSM(km)"]
rs = 58232

r_fill=100
th_fill = np.arange(24)*2*np.pi/24

th_c = np.arange(100)*2*np.pi/100

x_fill=r_fill*np.cos(th_fill)
y_fill=r_fill*np.sin(th_fill)

plt.figure()

for i in range(0,len(x_fill),2):
    plt.fill([0,x_fill[i],x_fill[i+1]],[0,y_fill[i],y_fill[i+1]],color="grey", alpha = 0.1)

for i in range(10,100,10):
    rc = i
    
    xc = rc*np.cos(th_c)
    yc = rc*np.sin(th_c)
    
    plt.plot(xc,yc, color = 'k', linestyle = (0, (1, 10)), linewidth = 1)

plt.scatter(x[starts[big_fps]+3071518]/rs,y[starts[big_fps]+3071518]/rs,s=50, color = 'orange', 
            label = 'False Detections')
plt.scatter(0,0, color='k', s=500.0, marker = 'o', label = "Saturn")
plt.ylim([-20,70])
plt.xlim([-30,30])
plt.xlabel('X/R$_S$', fontsize = 20)
plt.ylabel('Y/R$_S$', fontsize = 20)
plt.title('Detections of ML algorithm', fontsize = 20)
plt.legend(fontsize = 16)

minB_tps = np.array([])
minB_fps = np.array([])
minB_tot = np.array([])
minB_fns = np.array([])

for i in range(len(smith_starts)):
    if np.max(pred[s_s_2010[i]:s_e_2010[i]]) == 0:
        minB_fns = np.append(minB_fns, np.min(np.abs(Cass_data["BTotal(nT)"].iloc[s_2010[i]:e_2010[i]])))
 
for i in range(len(starts)):
    minB_tot = np.append(minB_tot, np.min(np.abs(Cass_data["BTotal(nT)"].iloc[s_2010[i]:e_2010[i]])))
    
    if TPS[i] == 1:
        minB_tps = np.append(minB_tps, np.min(np.abs(Cass_data["BTotal(nT)"].iloc[s_2010[i]:e_2010[i]])))
    else:
        minB_fps = np.append(minB_fps, np.min(np.abs(Cass_data["BTotal(nT)"].iloc[s_2010[i]:e_2010[i]])))

plt.figure()

plt.hist(minB_fns, bins = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0], color = 'red', label = "FN")
plt.hist(minB_tot, bins = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0], color = 'orange', label = "TP+FP")
plt.hist(minB_tps, bins = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0], color = 'green', label = "TP")
plt.hist(minB_fns, bins = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0], color = 'red')

plt.hist(minB_fps, bins = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0], color = 'red')


plt.xlabel("$Min B_\total\ (nT)$", fontsize = 20)
plt.ylabel('Counts', fontsize = 20)
plt.title("Distributions of Minimum Magnetic Field Strength", fontsize = 20)
plt.yscale('log')
plt.legend(fontsize = 16)

plt.show()


dur_bins = np.arange(0,60,2)
dur = ends - starts
s_dur = s_e_2010 - s_s_2010

plt.figure()

plt.hist(dur, bins = dur_bins, color = 'red', label = "ML durations")
plt.hist(s_dur, bins = dur_bins, color = 'orange', label = "Smith durations")

plt.xlabel("Duration (mins)$", fontsize = 20)
plt.ylabel('Counts', fontsize = 20)
plt.title("Distributions of Event Duration", fontsize = 20)
plt.yscale('log')
plt.xscale('log')

plt.legend(fontsize = 16)

plt.show()


dat = pd.read_csv(dir_path + "/MP_BS_Crossings.txt")

location = np.zeros(Cass_data.shape[0])+2
SCAS = 0
C_dats=Cass_data.index
date = []

for i in range(len(crossings)):
    YR = str(crossings['Year'][i])
    DOY = str(crossings['DOY'][i])
    HR = str(crossings['Hour'][i])
    MIN = str(crossings['Minute'][i])
    
    date = np.append(date, datetime.strptime(YR+' '+DOY+' '+HR+' '+MIN, '%Y %j %H %M'))
    
    if crossings['TYPE_DIRECTION'][i] == 'BSI':
        location[np.where(C_dats == str(date[i].year)+'-'+str(date[i].month)+'-'+str(date[i].day)+' '+str(date[i].hour)+':'+str(date[i].minute)+':30')[0][0]:]=1
    elif crossings['TYPE_DIRECTION'][i] == 'BSO':
        location[np.where(C_dats == str(date[i].year)+'-'+str(date[i].month)+'-'+str(date[i].day)+' '+str(date[i].hour)+':'+str(date[i].minute)+':30')[0][0]:]=2
    elif crossings['TYPE_DIRECTION'][i] == 'MPI':
        location[np.where(C_dats == str(date[i].year)+'-'+str(date[i].month)+'-'+str(date[i].day)+' '+str(date[i].hour)+':'+str(date[i].minute)+':30')[0][0]:]=0
    elif crossings['TYPE_DIRECTION'][i] == 'MPO':
        location[np.where(C_dats == str(date[i].year)+'-'+str(date[i].month)+'-'+str(date[i].day)+' '+str(date[i].hour)+':'+str(date[i].minute)+':30')[0][0]:]=1
    else:
        print(SCAS)
        SCAS+=1


colr=np.array([])
for i in range(len(big_fps)):
    colr=np.append(colr,np.max(location[(starts[big_fps[i]]+3071458):(starts[big_fps[i]]+3071578)]))

Tail=np.where(colr == 0)[0]
BowS=np.where(colr == 1)[0]
SW=np.where(colr == 2)[0]

plt.figure()

for i in range(0,len(x_fill),2):
    plt.fill([0,x_fill[i],x_fill[i+1]],[0,y_fill[i],y_fill[i+1]],color="grey", alpha = 0.1)

for i in range(10,100,10):
    rc = i
    
    xc = rc*np.cos(th_c)
    yc = rc*np.sin(th_c)
    
    plt.plot(xc,yc, color = 'k', linestyle = (0, (1, 10)), linewidth = 1)

plt.scatter(x[starts[big_fps[Tail]]+3071518]/rs,y[starts[big_fps[Tail]]+3071518]/rs,s=50, color = 'g', 
            label = 'MSphere detections')
plt.scatter(x[starts[big_fps[BowS]]+3071518]/rs,y[starts[big_fps[BowS]]+3071518]/rs,s=50, color = 'orange', 
            label = 'MSheathe detections')
plt.scatter(x[starts[big_fps[SW]]+3071518]/rs,y[starts[big_fps[SW]]+3071518]/rs,s=50, color = 'r', 
            label = 'SW detections')
plt.scatter(0,0, color='k', s=500.0, marker = 'o', label = "Saturn")
plt.ylim([-20,70])
plt.xlim([-30,30])
plt.xlabel('X/R$_S$', fontsize = 20)
plt.ylabel('Y/R$_S$', fontsize = 20)
plt.title('Detections of ML algorithm', fontsize = 20)
plt.legend(fontsize = 16)




colors=np.array(['g', 'orange', 'r'])

import matplotlib
import time

lt = (24.*p/360)

hours = lt.astype(int)
minutes = (lt*60) % 60

dat_index = r.index

for count in range(len(big_fps)):
    print(count)
    gridsize=(4,2)
    
    fig = plt.figure(figsize=(20,12))

    ax5 = plt.subplot2grid(gridsize,(0,0), rowspan = 4) 
    
    for i in range(0,len(x_fill),2):
        plt.fill([0,x_fill[i],x_fill[i+1]],[0,y_fill[i],y_fill[i+1]],color="grey", alpha = 0.1)
    for i in range(10,100,10):
        rc = i
        
        xc = rc*np.cos(th_c)
        yc = rc*np.sin(th_c)
        
        plt.plot(xc,yc, color = 'k', linestyle = (0, (1, 10)), linewidth = 1)
    
    plt.scatter(x[starts[big_fps[Tail]]+3071518]/rs,y[starts[big_fps[Tail]]+3071518]/rs,s=50, color = 'g', 
                label = 'MSphere detections')
    plt.scatter(x[starts[big_fps[BowS]]+3071518]/rs,y[starts[big_fps[BowS]]+3071518]/rs,s=50, color = 'orange', 
                label = 'MSheathe detections')
    plt.scatter(x[starts[big_fps[SW]]+3071518]/rs,y[starts[big_fps[SW]]+3071518]/rs,s=50, color = 'r', 
                label = 'SW detections')
    plt.scatter(x[starts[big_fps[count]]+3071518]/rs,y[starts[big_fps[count]]+3071518]/rs,s=500, color = colors[int(colr[count])], marker = 'X')
    plt.scatter(0,0, color='k', s=500.0, marker = 'o', label = "Saturn")
    plt.ylim([-20,70])
    plt.xlim([-30,30])
    plt.xlabel('X/R$_S$', fontsize = 20)
    plt.ylabel('Y/R$_S$', fontsize = 20)
    plt.title(str(Cass_data.index[starts[big_fps[count]]+3071518]), fontsize = 20)
    plt.text(-20,-13,'Nearest Crossing : '+str(date[np.argmin(np.abs(Cass_data.index[starts[big_fps[count]]+3071518]-date))]), fontsize = 18)
    plt.text(-20,-18,'No. of events within an hour : '+str(np.where(np.abs(starts-starts[big_fps[count]]) < 60)[0].shape[0]-1), fontsize = 18)

    plt.legend(fontsize = 16)
    
    plt.text(31.5,-29,'Hour (UT) \nRad (R$_S$) \nLat ($^\circ$) \nLT (hr:min)', fontsize = 13)
    

    ax4 = plt.subplot2grid(gridsize,(0,1))
    plt.title("Cassini Magnetic Measurements", fontsize = 20)
    ax1 = plt.subplot2grid(gridsize,(1,1))
    ax2 = plt.subplot2grid(gridsize,(2,1))
    ax3 = plt.subplot2grid(gridsize,(3,1))
    
    offs = 3071519
    midpoint = int(offs + (starts[big_fps[count]]+ends[big_fps[count]])/2.0)
    
    ax4.plot(Cass_data["BTotal(nT)"][midpoint - 60 : midpoint + 60])
    
    ax4.axvspan(Cass_data.index[offs + starts[big_fps[count]]], 
        Cass_data.index[offs + ends[big_fps[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
    
    
            
    ax4lim = ax4.get_ylim()
    ax4.set_ylabel("$|B|\ (nT)$", fontsize = 20)
    ax4.set_xticklabels([])
    ax4.legend()
    
    ax1.plot(Cass_data["BX_KRTP(nT)"][midpoint - 60 : midpoint + 60])
    ax1lim = ax1.get_ylim()
    ax1.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':',color='k')
    
    ax1.axvspan(Cass_data.index[offs + starts[big_fps[count]]], 
        Cass_data.index[offs + ends[big_fps[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
            
    ax1.set_ylabel("$B_r\ (nT)$", fontsize = 20)
    ax1.set_xticklabels([])
    ax1.legend()
    
    ax2.plot(Cass_data["BY_KRTP(nT)"][midpoint - 60 : midpoint + 60])
    ax2lim = ax2.get_ylim()
    ax2.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':', color='k')

    
    ax2.axvspan(Cass_data.index[offs + starts[big_fps[count]]], 
        Cass_data.index[offs + ends[big_fps[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
            
    ax2.set_ylabel("$B_\\theta\ (nT)$", fontsize = 20)
    ax2.set_xticklabels([])
    ax2.legend()
    
    ax3.plot(Cass_data["BZ_KRTP(nT)"][midpoint - 60 : midpoint + 60])
    ax3lim = ax3.get_ylim()
    ax3xlim = ax3.get_xlim()[0]
    ax3.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':', color='k')

    ax3.axvspan(Cass_data.index[offs + starts[big_fps[count]]], 
        Cass_data.index[offs + ends[big_fps[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
            
    ax3.set_ylabel("$B_\\phi\ (nT)$", fontsize = 20)
    ax3.legend()
    
    ymin=np.min([ax1lim[0],ax2lim[0],ax3lim[0]])
    ymax=np.max([ax1lim[1],ax2lim[1],ax3lim[1]])
    
    ax4.set_ylim([0,ax4lim[1]])
    ax1.set_ylim([ymin,ymax])
    ax2.set_ylim([ymin,ymax])
    ax3.set_ylim([ymin,ymax])
    
    date_form = matplotlib.dates.DateFormatter("%H:%M")
    ax3.xaxis.set_major_formatter(date_form)

    fig.canvas.draw()
    labels = [item.get_text() for item in ax3.get_xticklabels()]
    
    date_form = matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M")
    ax3.xaxis.set_major_formatter(date_form)
    
    ax3.set_ylim([ymin,ymax])

    fig.canvas.draw()
    label_dates = [item.get_text() for item in ax3.get_xticklabels()]
    
    for i in range(len(labels)):
        tick_date = np.where(dat_index == (label_dates[i]+':30'))[0][0]
        labels[i] = (labels[i] + '\n'+str(round(r.iloc[tick_date],2)) + '\n'+str(round(t.iloc[tick_date],2)) + '\n'+str(hours[tick_date])+':'+str(minutes[tick_date].astype(int)).rjust(2,'0'))
    
    ax3.set_xticklabels(labels)
    
    ax4.tick_params(axis='y', which='major', labelsize=14)
    ax1.tick_params(axis='y', which='major', labelsize=14)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    ax3.set_ylim([ymin,ymax])
        
    fig.savefig(dir_path+'\\Images\\Large_Bth_FPS\\Event_'+str(count)+'.png')
    plt.close(fig)




mid = (3071519+((starts+ends)/2.)).astype(int)

ev_lt = lt[Cass_data.index[mid]]
day = np.where((ev_lt > 6.0) & (ev_lt < 18.0))[0]
night = np.where((ev_lt < 6.0) | (ev_lt > 18.0))[0]

ev_list_r = [Cass_data["BX_KRTP(nT)"][mid[0] - 30 : mid[0] + 30]]
ev_list_th = [Cass_data["BY_KRTP(nT)"][mid[0] - 30 : mid[0] + 30]]
ev_list_ph = [Cass_data["BZ_KRTP(nT)"][mid[0] - 30 : mid[0] + 30]]


for i in mid[1:]:
    ev_list_r=np.concatenate((ev_list_r, [Cass_data["BX_KRTP(nT)"][i - 30 : i + 30]]), axis = 0)
    ev_list_th=np.concatenate((ev_list_th, [Cass_data["BY_KRTP(nT)"][i - 30 : i + 30]]), axis = 0)
    ev_list_ph=np.concatenate((ev_list_ph, [Cass_data["BZ_KRTP(nT)"][i - 30 : i + 30]]), axis = 0)
    

Cass_index = Cass_data.index
smith_dates = pd.to_datetime(smith_catalogue["EventDateTime"], format = '%d/%m/%Y %H:%M')+pd.Timedelta(seconds=30)
cent_ind = np.where(Cass_index == smith_dates[0])[0][0]
smith_B_th = [Cass_data["BY_KRTP(nT)"][cent_ind - 30 : cent_ind + 30]]

for i in smith_dates[1:]:
    cent_ind = np.where(Cass_index == i)[0][0]
    smith_B_th = np.concatenate((smith_B_th, [Cass_data["BY_KRTP(nT)"][cent_ind - 30 : cent_ind + 30]]), axis = 0)

dayside_r = np.mean(np.abs(ev_list_r[day,:]),axis = 0)
nightside_r = np.mean(np.abs(ev_list_r[night,:]),axis = 0)
dayside_th = np.mean(ev_list_th[day,:],axis = 0)
nightside_th = np.mean(ev_list_th[night,:],axis = 0)
dayside_ph = np.mean(np.abs(ev_list_ph[day,:]),axis = 0)
nightside_ph = np.mean(np.abs(ev_list_ph[night,:]),axis = 0)

smith_th = np.mean(smith_B_th, axis = 0)

t_diff = (np.arange(60)-30)

plt.figure()

for i in range(len(ev_list_th)):
    plt.plot(t_diff, ev_list_th[i,:], color = 'g', linewidth=0.1)

plt.plot(t_diff, dayside_th, color = 'c', linewidth=2, label = 'Day-side detections')
plt.plot(t_diff, nightside_th, color = 'midnightblue', linewidth=2, label = 'Night-side detections')
plt.plot(t_diff, smith_th, color = 'k', linewidth=2, label = '{} Smith et al, 2016. detections'.format(smith_B_th.shape[0]))
plt.plot(t_diff,np.zeros(len(t_diff)), linestyle = (0, (5, 10)), color = 'k')

plt.ylim([-1.0,3.0])
plt.xlabel('T - T$_0$ (min)', fontsize = 20)
plt.ylabel('$B_\\theta\ (nT)$', fontsize = 20)
plt.title('Epoch analysis of {} day-side and {} night-side detections'.format(day.shape[0], night.shape[0]), fontsize = 20, pad = 20)

plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

plt.legend(fontsize = 20)

plt.show()




smith_starts = pd.to_datetime(smith_catalogue["EventStartDateTime"], format = '%d/%m/%Y %H:%M')+pd.Timedelta(seconds=30)
smith_ends = pd.to_datetime(smith_catalogue["EventEndDateTime"], format = '%d/%m/%Y %H:%M')+pd.Timedelta(seconds=30)

smith_dir = Cass_data["BY_KRTP(nT)"][smith_starts].values - Cass_data["BY_KRTP(nT)"][smith_ends].values
smith_tailward = np.where((smith_dir > 0) & (smith_catalogue["Class"].values == 'Plasmoid'))[0]


cent_ind = np.where(Cass_index == smith_dates[smith_tailward[0]])[0][0]
smith_B_th = [Cass_data["BY_KRTP(nT)"][cent_ind - 30 : cent_ind + 30]]

for i in smith_dates[smith_tailward[1:]]:
    cent_ind = np.where(Cass_index == i)[0][0]
    smith_B_th = np.concatenate((smith_B_th, [Cass_data["BY_KRTP(nT)"][cent_ind - 30 : cent_ind + 30]]), axis = 0)


max_loc = []
min_loc = []

for i in range(len(starts)):
    max_loc = np.append(max_loc, np.argmax(Cass_data["BY_KRTP(nT)"][starts[i]+3071519:ends[i]+3071519+1].values))
    min_loc = np.append(min_loc, np.argmin(Cass_data["BY_KRTP(nT)"][starts[i]+3071519:ends[i]+3071519+1].values))

direction = max_loc-min_loc
ev_tailward = np.where(direction < 0)[0]


day = np.where((ev_lt > 6.0) & (ev_lt < 18.0) & (direction < 0))[0]
night = np.where(np.logical_and(np.logical_or(ev_lt < 6.0, ev_lt > 18.0), direction < 0))[0]

dayside_th = np.mean(ev_list_th[day,:],axis = 0)
nightside_th = np.mean(ev_list_th[night,:],axis = 0)

smith_th = np.mean(smith_B_th, axis = 0)

t_diff = (np.arange(60)-30)

plt.figure()

for i in np.where(direction < 0)[0]:
    plt.plot(t_diff, ev_list_th[i,:], color = 'g', linewidth=0.1)

plt.plot(t_diff, dayside_th, color = 'c', linewidth=2, label = 'Day-side detections')
plt.plot(t_diff, nightside_th, color = 'midnightblue', linewidth=2, label = 'Night-side detections')
plt.plot(t_diff, smith_th, color = 'k', linewidth=2, label = '{} Smith et al, 2016. detections'.format(smith_B_th.shape[0]))
plt.plot(t_diff,np.zeros(len(t_diff)), linestyle = (0, (5, 10)), color = 'k')

plt.ylim([-1.0,3.0])
plt.xlabel('T - T$_0$ (min)', fontsize = 20)
plt.ylabel('$B_\\theta\ (nT)$', fontsize = 20)
plt.title('Epoch analysis of {} day-side and {} night-side tailward detections'.format(day.shape[0], night.shape[0]), fontsize = 20, pad =20)

plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

plt.legend(fontsize = 20)

plt.show()




file = open('testfile.txt','w') 

for i in range(len(np.where(ml_events==1)[0])):
    file.write(str(np.where(ml_events == 1)[0][i]+3071519)+', ')

file.close

plt.figure()

plt.plot(Cass_data["BY_KRTP(nT)"][3071519:3597118])

for i in range(np.nonzero(starts)[0].shape[0]):
    if i == 0:
        plt.axvspan(Cass_data.index[starts[i]+3071519], 
        Cass_data.index[ends[i]+3071519], 
        color = 'red', alpha = 0.5, label = 'ML Event')
    
    else:
        plt.axvspan(Cass_data.index[starts[i]+3071519], 
        Cass_data.index[ends[i]+3071519], 
        color = 'red', alpha = 0.5)
        

y1 = unsearched_events[3071519:3597118] > 0.5
y1_shifted = np.roll(y1,1)
starts_cat = y1 & ~y1_shifted
ends_cat = ~y1 & y1_shifted

starts_cat = np.nonzero(starts_cat)[0]
ends_cat = np.nonzero(ends_cat)[0]

for i in range(np.nonzero(starts_cat)[0].shape[0]):
    if i == 0:
        plt.axvspan(Cass_data.index[starts_cat[i]+3071519], 
        Cass_data.index[ends_cat[i]+3071519], 
        color = 'blue', alpha = 0.5, label = 'Andy Event')
    
    else:
        plt.axvspan(Cass_data.index[starts_cat[i]+3071519], 
        Cass_data.index[ends_cat[i]+3071519], 
        color = 'blue', alpha = 0.5)
        
plt.ylabel("$B_\\theta\ (nT)$")
plt.legend()


unsearched_events[np.where(unsearched_events > 1)[0]] = 1

obs = unsearched_events

obs_b = obs > 0.5
obs_shifted = np.roll(obs_b,1)
smith_starts = obs_b & ~obs_shifted
smith_ends = ~obs_b & obs_shifted
smith_starts = np.nonzero(smith_starts)[0]
smith_ends = np.nonzero(smith_ends)[0]



iden=1096
gridsize=(4,1)
    
fig = plt.figure(figsize=(10,12))

midpoint = int((smith_starts[iden]+smith_ends[iden])/2.0)
 
ax4 = plt.subplot2grid(gridsize,(3,0))
ax1 = plt.subplot2grid(gridsize,(0,0))
plt.title("Cassini Magnetic Measurements : " + str(Cass_data.index[midpoint]), fontsize = 20)
ax2 = plt.subplot2grid(gridsize,(1,0))
ax3 = plt.subplot2grid(gridsize,(2,0))
    
offs = 3071519

ax4.plot(Cass_data["BTotal(nT)"][midpoint - 60 : midpoint + 60])
    
ax4.axvspan(Cass_data.index[smith_starts[iden]], 
    Cass_data.index[smith_ends[iden]], 
    color = 'red', alpha = 0.5, label = 'Plasmoid Event')
        
ax4lim = ax4.get_ylim()
ax4.set_ylabel("$|B|\ (nT)$", fontsize = 20)
ax4.set_xticklabels([])
ax4.legend()

plt.text(Cass_data.index[midpoint - 83],-2.75,'Hour (UT) \nRad (R$_S$) \nLat ($^\circ$) \nLT (hr:min)', fontsize = 13)

ax1.plot(Cass_data["BX_KRTP(nT)"][midpoint - 60 : midpoint + 60])
ax1lim = ax1.get_ylim()
ax1.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':',color='k')
    
ax1.axvspan(Cass_data.index[smith_starts[iden]], 
    Cass_data.index[smith_ends[iden]], 
    color = 'red', alpha = 0.5, label = 'ML Event')
        
ax1.set_ylabel("$B_r\ (nT)$", fontsize = 20)
ax1.set_xticklabels([])
    
ax2.plot(Cass_data["BY_KRTP(nT)"][midpoint - 60 : midpoint + 60])
ax2lim = ax2.get_ylim()
ax2.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':', color='k')

    
ax2.axvspan(Cass_data.index[smith_starts[iden]], 
    Cass_data.index[smith_ends[iden]], 
    color = 'red', alpha = 0.5, label = 'ML Event')
            
ax2.set_ylabel("$B_\\theta\ (nT)$", fontsize = 20)
ax2.set_xticklabels([])
    
ax3.plot(Cass_data["BZ_KRTP(nT)"][midpoint - 60 : midpoint + 60])
ax3lim = ax3.get_ylim()
ax3xlim = ax3.get_xlim()[0]
ax3.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':', color='k')

ax3.axvspan(Cass_data.index[smith_starts[iden]], 
    Cass_data.index[smith_ends[iden]], 
    color = 'red', alpha = 0.5, label = 'ML Event')
            
ax3.set_ylabel("$B_\\phi\ (nT)$", fontsize = 20)
    
ymin=np.min([ax1lim[0],ax2lim[0],ax3lim[0]])
ymax=np.max([ax1lim[1],ax2lim[1],ax3lim[1]])
    
ax4.set_ylim([0,ax4lim[1]])
ax1.set_ylim([ymin,ymax])
ax2.set_ylim([ymin,ymax])
ax3.set_ylim([ymin,ymax])
    
date_form = matplotlib.dates.DateFormatter("%H:%M")
ax3.xaxis.set_major_formatter(date_form)

fig.canvas.draw()
labels = [item.get_text() for item in ax3.get_xticklabels()]
    
date_form = matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M")
ax3.xaxis.set_major_formatter(date_form)
    
ax3.set_ylim([ymin,ymax])

fig.canvas.draw()
label_dates = [item.get_text() for item in ax3.get_xticklabels()]
    
for i in range(len(labels)):
    tick_date = np.where(dat_index == (label_dates[i]+':30'))[0][0]
    labels[i] = (labels[i] + '\n'+str(round(r.iloc[tick_date],2)) + '\n'+str(round(t.iloc[tick_date],2)) + '\n'+str(hours[tick_date])+':'+str(minutes[tick_date].astype(int)).rjust(2,'0'))
    
ax3.set_xticklabels(labels)
    
ax4.tick_params(axis='y', which='major', labelsize=14)
ax1.tick_params(axis='y', which='major', labelsize=14)
ax2.tick_params(axis='y', which='major', labelsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
    
ax3.set_ylim([ymin,ymax])        


    
fig = plt.figure(figsize=(10,12))

midpoint = 3250000
 
ax4 = plt.subplot2grid(gridsize,(3,0))
ax1 = plt.subplot2grid(gridsize,(0,0))
plt.title("Cassini Magnetic Measurements : " + str(Cass_data.index[midpoint]), fontsize = 20)
ax2 = plt.subplot2grid(gridsize,(1,0))
ax3 = plt.subplot2grid(gridsize,(2,0))
    
offs = 3071519

ax4.plot(Cass_data["BTotal(nT)"][midpoint - 60 : midpoint + 60])
     
ax4lim = ax4.get_ylim()
ax4.set_ylabel("$|B|\ (nT)$", fontsize = 20)
ax4.set_xticklabels([])

plt.text(Cass_data.index[midpoint - 83],-5.5,'Hour (UT) \nRad (R$_S$) \nLat ($^\circ$) \nLT (hr:min)', fontsize = 13)


ax1.plot(Cass_data["BX_KRTP(nT)"][midpoint - 60 : midpoint + 60])
ax1lim = ax1.get_ylim()
ax1.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':',color='k')

ax1.set_ylabel("$B_r\ (nT)$", fontsize = 20)
ax1.set_xticklabels([])
    
ax2.plot(Cass_data["BY_KRTP(nT)"][midpoint - 60 : midpoint + 60])
ax2lim = ax2.get_ylim()
ax2.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':', color='k')
           
ax2.set_ylabel("$B_\\theta\ (nT)$", fontsize = 20)
ax2.set_xticklabels([])
    
ax3.plot(Cass_data["BZ_KRTP(nT)"][midpoint - 60 : midpoint + 60])
ax3lim = ax3.get_ylim()
ax3xlim = ax3.get_xlim()[0]
ax3.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':', color='k')
           
ax3.set_ylabel("$B_\\phi\ (nT)$", fontsize = 20)
    
ymin=np.min([ax1lim[0],ax2lim[0],ax3lim[0]])
ymax=np.max([ax1lim[1],ax2lim[1],ax3lim[1]])
    
ax4.set_ylim([0,ax4lim[1]])
ax1.set_ylim([ymin,ymax])
ax2.set_ylim([ymin,ymax])
ax3.set_ylim([ymin,ymax])
    
date_form = matplotlib.dates.DateFormatter("%H:%M")
ax3.xaxis.set_major_formatter(date_form)

fig.canvas.draw()
labels = [item.get_text() for item in ax3.get_xticklabels()]
    
date_form = matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M")
ax3.xaxis.set_major_formatter(date_form)
    
ax3.set_ylim([ymin,ymax])

fig.canvas.draw()
label_dates = [item.get_text() for item in ax3.get_xticklabels()]
    
for i in range(len(labels)):
    tick_date = np.where(dat_index == (label_dates[i]+':30'))[0][0]
    labels[i] = (labels[i] + '\n'+str(round(r.iloc[tick_date],2)) + '\n'+str(round(t.iloc[tick_date],2)) + '\n'+str(hours[tick_date])+':'+str(minutes[tick_date].astype(int)).rjust(2,'0'))
    
ax3.set_xticklabels(labels)
    
ax4.tick_params(axis='y', which='major', labelsize=14)
ax1.tick_params(axis='y', which='major', labelsize=14)
ax2.tick_params(axis='y', which='major', labelsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
    
ax3.set_ylim([ymin,ymax])

        

"""
"""

from matplotlib.patches import ConnectionPatch
import matplotlib.dates as mdates

gridsize=(3,1)
fmt = mdates.DateFormatter('%Y-%j')

fig = plt.figure(figsize=(25,25))

ax1 = plt.subplot2grid(gridsize,(0,0))
plt.title("Cassini Magnetic Measurements", fontsize = 20)
ax2 = plt.subplot2grid(gridsize,(1,0))
ax3 = plt.subplot2grid(gridsize,(2,0))


ax1.plot(Cass_data["BY_KRTP(nT)"][3185279:3467519],color = 'k')

for i in range(np.nonzero(starts)[0].shape[0]):
    if i == 0:
        ax1.fill([Cass_data.index[starts[i]+3071519], 
        Cass_data.index[ends[i]+3071519],Cass_data.index[ends[i]+3071519],Cass_data.index[starts[i]+3071519]],
        [0,0,-6,-6], color = 'red', alpha = 0.5)
    
    else:
        ax1.fill([Cass_data.index[starts[i]+3071519], 
        Cass_data.index[ends[i]+3071519],Cass_data.index[ends[i]+3071519],Cass_data.index[starts[i]+3071519]],
        [0,0,-6,-6],
        color = 'red', alpha = 0.5)
    

y1 = unsearched_events > 0.5
y1_shifted = np.roll(y1,1)
starts_cat = y1 & ~y1_shifted
ends_cat = ~y1 & y1_shifted

starts_cat = np.nonzero(starts_cat)[0]
ends_cat = np.nonzero(ends_cat)[0]

for i in range(np.nonzero(starts_cat)[0].shape[0]):
    if i == 0:
        ax1.fill([Cass_data.index[starts_cat[i]], 
        Cass_data.index[ends_cat[i]],Cass_data.index[ends_cat[i]],Cass_data.index[starts_cat[i]]],
        [0,0,6,6], 
        color = 'blue', alpha = 0.5, label = 'Smith Event')
    
    else:
        ax1.fill([Cass_data.index[starts_cat[i]], 
        Cass_data.index[ends_cat[i]],Cass_data.index[ends_cat[i]],Cass_data.index[starts_cat[i]]],
        [0,0,6,6], 
        color = 'blue', alpha = 0.5)
        
ax1.plot([Cass_data.index[3185279],Cass_data.index[3467519]],[0,0],linestyle = '--', color = 'k')

ax1.set_ylabel("$B_\\theta\ (nT)$", fontsize = 20)
ax1.set_xlim([Cass_data.index[3185279],Cass_data.index[3467519]])
ax1.set_ylim([-4,6])
ax1.tick_params(labelsize=18)
#ax1.legend()
ax1.xaxis.set_major_formatter(fmt)

ax1.plot([Cass_data.index[3333000],Cass_data.index[3333000]],[6,-4],color = 'k', linewidth = 5)
ax1.plot([Cass_data.index[3376000],Cass_data.index[3376000]],[6,-4],color = 'k', linewidth = 5)


ax2.plot(Cass_data["BY_KRTP(nT)"][3333000:3376000],color = 'k')

for i in range(np.nonzero(starts)[0].shape[0]):
    if i == 0:
        ax2.fill([Cass_data.index[starts[i]+3071519], 
        Cass_data.index[ends[i]+3071519],Cass_data.index[ends[i]+3071519],Cass_data.index[starts[i]+3071519]],
        [0,0,-6,-6], 
        color = 'red', alpha = 0.5)
    
    else:
        ax2.fill([Cass_data.index[starts[i]+3071519], 
        Cass_data.index[ends[i]+3071519],Cass_data.index[ends[i]+3071519],Cass_data.index[starts[i]+3071519]],
        [0,0,-6,-6], 
        color = 'red', alpha = 0.5)
        

for i in range(np.nonzero(starts_cat)[0].shape[0]):
    if i == 0:
        ax2.fill([Cass_data.index[starts_cat[i]], 
        Cass_data.index[ends_cat[i]],Cass_data.index[ends_cat[i]],Cass_data.index[starts_cat[i]]],
        [0,0,6,6], 
        color = 'blue', alpha = 0.5, label = 'Smith Event')
    
    else:
        ax2.fill([Cass_data.index[starts_cat[i]], 
        Cass_data.index[ends_cat[i]],Cass_data.index[ends_cat[i]],Cass_data.index[starts_cat[i]]],
        [0,0,6,6], 
        color = 'blue', alpha = 0.5)
        
ax2.plot([Cass_data.index[3333000],Cass_data.index[3376000]],[0,0],linestyle = '--', color = 'k')

ax2.set_ylabel("$B_\\theta\ (nT)$", fontsize = 20)
ax2.set_xlim([Cass_data.index[3333000],Cass_data.index[3376000]])
ax2.set_ylim([-4,6])
ax2.tick_params(labelsize=18)
#ax2.legend()
ax2.xaxis.set_major_formatter(fmt)

con = ConnectionPatch(xyA=(Cass_data.index[3333000],6), xyB=(Cass_data.index[3333000],-4), coordsA="data", coordsB="data",
                      axesA=ax2, axesB=ax1, color="k", linewidth = 4)
ax2.add_artist(con)
con = ConnectionPatch(xyA=(Cass_data.index[3376000],6), xyB=(Cass_data.index[3376000],-4), coordsA="data", coordsB="data",
                      axesA=ax2, axesB=ax1, color="k", linewidth = 4)
ax2.add_artist(con)

ax2.plot([Cass_data.index[3333000],Cass_data.index[3333000]],[6,-4],color = 'k', linewidth = 8)
ax2.plot([Cass_data.index[3376000],Cass_data.index[3376000]],[6,-4],color = 'k', linewidth = 8)

ax2.plot([Cass_data.index[3363500],Cass_data.index[3363500]],[6,-4],color = 'k', linewidth = 5)
ax2.plot([Cass_data.index[3364000],Cass_data.index[3364000]],[6,-4],color = 'k', linewidth = 5)





ax3.plot(Cass_data["BY_KRTP(nT)"][3363500:3364000],color = 'k')

for i in range(np.nonzero(starts)[0].shape[0]):
    if i == 0:
        ax3.fill([Cass_data.index[starts[i]+3071519], 
        Cass_data.index[ends[i]+3071519],Cass_data.index[ends[i]+3071519],Cass_data.index[starts[i]+3071519]],
        [0,0,-6,-6], 
        color = 'red', alpha = 0.5, label = 'ML Event')
    
    else:
        ax3.fill([Cass_data.index[starts[i]+3071519], 
        Cass_data.index[ends[i]+3071519],Cass_data.index[ends[i]+3071519],Cass_data.index[starts[i]+3071519]],
        [0,0,-6,-6], 
        color = 'red', alpha = 0.5)
        

for i in range(np.nonzero(starts_cat)[0].shape[0]):
    if i == 0:
        ax3.fill([Cass_data.index[starts_cat[i]], 
        Cass_data.index[ends_cat[i]],Cass_data.index[ends_cat[i]],Cass_data.index[starts_cat[i]]],
        [0,0,6,6], 
        color = 'blue', alpha = 0.5, label = 'S16 Event')
    
    else:
        ax3.fill([Cass_data.index[starts_cat[i]], 
        Cass_data.index[ends_cat[i]],Cass_data.index[ends_cat[i]],Cass_data.index[starts_cat[i]]],
        [0,0,6,6],
        color = 'blue', alpha = 0.5)
        
ax3.plot([Cass_data.index[3363500],Cass_data.index[3364000]],[0,0],linestyle = '--', color = 'k')

ax3.set_ylabel("$B_\\theta\ (nT)$", fontsize = 20)
ax3.set_xlabel("Date (YEAR-DOY)", fontsize = 20)
ax3.set_xlim([Cass_data.index[3363500],Cass_data.index[3364000]])
ax3.set_ylim([-2,3])
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
ax3.legend(fontsize = 20)
ax3.xaxis.set_major_formatter(fmt)

con = ConnectionPatch(xyA=(Cass_data.index[3363500],3), xyB=(Cass_data.index[3363500],-4), coordsA="data", coordsB="data",
                      axesA=ax3, axesB=ax2, color="k", linewidth = 4)
ax3.add_artist(con)
con = ConnectionPatch(xyA=(Cass_data.index[3364000],3), xyB=(Cass_data.index[3364000],-4), coordsA="data", coordsB="data",
                      axesA=ax3, axesB=ax2, color="k", linewidth = 4)
ax3.add_artist(con)


ax3.plot([Cass_data.index[3363500],Cass_data.index[3363500]],[3,-2],color = 'k', linewidth = 8)
ax3.plot([Cass_data.index[3364000],Cass_data.index[3364000]],[3,-2],color = 'k', linewidth = 8)



theta = 2*np.pi*(np.arange(360)-180)/360

a_1 = 10.3
a_2 = 0.2
a_3 = 0.73
a_4 =0.4

D_p = 0.01

r_0 = a_1*(D_p**(-a_2))
K = a_3 + a_4*D_p
rb1 = r_0*((2/(1+np.cos(theta)))**K)

D_p = 0.1

r_0 = a_1*(D_p**(-a_2))
K = a_3 + a_4*D_p
rb2 = r_0*((2/(1+np.cos(theta)))**K)

xb1 = rb1*np.cos(theta)
yb1 = rb1*np.sin(theta)

xb2 = rb2*np.cos(theta)
yb2 = rb2*np.sin(theta)

low_alt = np.where(rb2 < 75)

non_eligible_magneto_x = xb2[low_alt]
non_eligible_magneto_y = yb2[low_alt]

non_eligible_magneto_x = np.append(non_eligible_magneto_x,([75,75]))
non_eligible_magneto_y = np.append(non_eligible_magneto_y,([75,-75]))

non_eligible_day_x = ([0,0,75,75])
non_eligible_day_y = ([-75,75,75,-75])

non_eligible_lowr_x = 15*np.cos(theta)
non_eligible_lowr_y = 15*np.sin(theta)

saturn_x = 1*np.cos(theta)
saturn_y = 1*np.sin(theta)

Cass_data_ksm = pd.read_csv(dir_path + "/Cassini_Observations/Cass_data_ksm.txt", 
                        sep = "\t", header = 5008)

Cass_data_ksm = Cass_data_ksm.drop(np.arange(83386))

Cass_data_ksm["Timestamp(UTC)"] = pd.to_datetime(Cass_data_ksm["Timestamp(UTC)"], 
         format="%d/%m/%Y %H:%M:%S.%f")
Cass_data_ksm = Cass_data_ksm.set_index("Timestamp(UTC)")
Cass_data_ksm = Cass_data_ksm.resample("1T",loffset="30s").asfreq()

x = Cass_data_ksm["X_KSM(km)"]
y = Cass_data_ksm["Y_KSM(km)"]
z = Cass_data_ksm["Z_KSM(km)"]

ten=np.where((Cass_data.index < datetime(2011, 1, 1)) & (Cass_data.index > datetime(2010, 1, 1)))[0]

ten_elig_iloc = np.where(np.logical_and(np.logical_and(location[ten] == 0, r[ten] >=15), np.logical_or(lt[ten] <= 6, lt[ten] >= 18)))[0]

x_ten_inelig = x[ten]/rs
y_ten_inelig = y[ten]/rs

x_ten_inelig[ten_elig_iloc] = np.nan
y_ten_inelig[ten_elig_iloc] = np.nan

plt.figure(figsize=(10,14))
#plt.fill(non_eligible_magneto_x, non_eligible_magneto_y, color ='red')
#plt.fill(non_eligible_day_x, non_eligible_day_y, color ='red')
#plt.fill(non_eligible_lowr_x, non_eligible_lowr_y, color ='red')

for i in range(0,len(x_fill),2):
    plt.fill([0,x_fill[i],x_fill[i+1]],[0,y_fill[i],y_fill[i+1]],color="grey", alpha = 0.1)
for i in range(10,100,10):
    rc = i
        
    xc = rc*np.cos(th_c)
    yc = rc*np.sin(th_c)
    
    plt.plot(xc,yc, color = 'k', linestyle = (0, (1, 10)), linewidth = 1)
 
#plt.plot(xb1,yb1,color = 'k', linestyle = '--', linewidth = 1)    
#plt.plot(xb2,yb2,color = 'k', linestyle = '--', linewidth = 1)


plt.plot(x[ten]/rs,y[ten]/rs, label = "Cassini 2010 trajectory", 
          color = "royalblue", linewidth = 1.0)  
plt.plot(x_ten_inelig,y_ten_inelig, label = "Spatially ineligible data", 
          color = "red", linewidth = 1.0)  

plt.fill(saturn_x,saturn_y, color='khaki', label = "Saturn")
plt.ylim([-10,50])
plt.xlim([-25,15])
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.xlabel('X/R$_S$', fontsize = 20)
plt.ylabel('Y/R$_S$', fontsize = 20)
plt.title("Spatially ineligible reconnection events for the Smith catalogue", fontsize = 20, pad = 20)


"""

#=====SVM=====#
"""

from sklearn.svm import SVC

x_train = np.reshape(Input_test_train,(Input_test_train.shape[0],
                               Input_test_train.shape[1]*Input_test_train.shape[2]))
y_train = np.where(Outp_test_train == 1)[1]

x_test = np.reshape(Input_val,(Input_val.shape[0],
                               Input_val.shape[1]*Input_val.shape[2]))
y_test = np.where(Outp_val == 1)[1]

model = SVC(C = 1.0, gamma = 100, kernel = 'linear')
model.fit(x_train, y_train)

score = model.score(x_test, y_test)

# Output the loss of the model's weights and biases and the accuracy of the model
# on the "score" dataset
print('SVM accuracy:', score)

"""





"""

bth_rms=[]
for i in mid:
    bth_rms = np.append(bth_rms, np.sqrt(np.mean(Cass_data["BY_KRTP(nT)"][i - 30 : i + 30]**2)))

possible = np.where((location[mid] == 0) & ((lt[mid] > 18) | (lt[mid] < 6)) & (r[mid] > 15) & (dbth > 0.25) & ((dbth/bth_rms)>=1.5))

obs = obs[3071519:3597118]

obs_b = obs > 0.5
obs_shifted = np.roll(obs_b,1)
smith_starts = obs_b & ~obs_shifted
smith_ends = ~obs_b & obs_shifted
smith_starts = np.nonzero(smith_starts)[0]
smith_ends = np.nonzero(smith_ends)[0]

Ev_num =[]
for i in range(len(starts)):
    Ev_num = np.append(Ev_num, np.max(obs[starts[i]:ends[i]]))

smith_ev_num =[]
for i in range(len(smith_starts)):
    smith_ev_num = np.append(smith_ev_num, np.max(pred[smith_starts[i]:smith_ends[i]]))


TPE = np.where(smith_ev_num == 1)[0].shape
FPE = np.where(Ev_num[possible] == 0)[0].shape
FNE = np.where(smith_ev_num == 0)[0].shape



TP_ind = np.where(smith_ev_num == 1)[0]
FN_ind = np.where(smith_ev_num == 0)[0]
FP_ind = possible[0][np.where(Ev_num[possible] == 0)[0]]

TP_dbth = []
TP_dur = []
TP_lt = []
TP_rad = []

for count in range(len(TP_ind)):
    TP_dbth = np.append(TP_dbth, np.max(Cass_data["BY_KRTP(nT)"].iloc[smith_starts[TP_ind[count]]+3071519:smith_ends[TP_ind[count]]+3071520]) - np.min(Cass_data["BY_KRTP(nT)"].iloc[smith_starts[TP_ind[count]]+3071519:smith_ends[TP_ind[count]]+3071520]))
    TP_dur = np.append(TP_dur, smith_ends[TP_ind[count]]-smith_starts[TP_ind[count]])
    TP_lt = np.append(TP_lt, lt[smith_starts[TP_ind[count]]+3071519])
    TP_rad = np.append(TP_rad, r[smith_starts[TP_ind[count]]+3071519])

FN_dbth = []
FN_dur = []
FN_lt = []
FN_rad = []

for count in range(len(FN_ind)):
    FN_dbth = np.append(FN_dbth, np.max(Cass_data["BY_KRTP(nT)"].iloc[smith_starts[FN_ind[count]]+3071519:smith_ends[FN_ind[count]]+3071520]) - np.min(Cass_data["BY_KRTP(nT)"].iloc[smith_starts[FN_ind[count]]+3071519:smith_ends[FN_ind[count]]+3071520]))
    FN_dur = np.append(FN_dur, smith_ends[FN_ind[count]]-smith_starts[FN_ind[count]])
    FN_lt = np.append(FN_lt, lt[smith_starts[FN_ind[count]]+3071519])
    FN_rad = np.append(FN_rad, r[smith_starts[FN_ind[count]]+3071519])
    
    print(count)
    gridsize=(4,2)
    
    fig = plt.figure(figsize=(20,12))

    ax5 = plt.subplot2grid(gridsize,(0,0), rowspan = 4) 
    
    for i in range(0,len(x_fill),2):
        plt.fill([0,x_fill[i],x_fill[i+1]],[0,y_fill[i],y_fill[i+1]],color="grey", alpha = 0.1)
    for i in range(10,100,10):
        rc = i
        
        xc = rc*np.cos(th_c)
        yc = rc*np.sin(th_c)
        
        plt.plot(xc,yc, color = 'k', linestyle = (0, (1, 10)), linewidth = 1)
    
    plt.scatter(x[smith_starts[FN_ind]+3071518]/rs,y[smith_starts[FN_ind]+3071518]/rs,s=50, color = 'r', 
                label = 'False Negative Events')
    plt.scatter(x[smith_starts[FN_ind[count]]+3071518]/rs,y[smith_starts[FN_ind[count]]+3071518]/rs,s=500, color = 'g', marker = 'X')
    plt.scatter(0,0, color='k', s=500.0, marker = 'o', label = "Saturn")
    plt.ylim([-20,70])
    plt.xlim([-30,30])
    plt.xlabel('X/R$_S$', fontsize = 20)
    plt.ylabel('Y/R$_S$', fontsize = 20)
    plt.title(str(Cass_data.index[smith_starts[FN_ind[count]]+3071518]), fontsize = 20)

    plt.legend(fontsize = 16)
    
    plt.text(31.5,-29,'Hour (UT) \nRad (R$_S$) \nLat ($^\circ$) \nLT (hr:min)', fontsize = 13)
    

    ax4 = plt.subplot2grid(gridsize,(0,1))
    plt.title("Cassini Magnetic Measurements", fontsize = 20)
    ax1 = plt.subplot2grid(gridsize,(1,1))
    ax2 = plt.subplot2grid(gridsize,(2,1))
    ax3 = plt.subplot2grid(gridsize,(3,1))
    
    offs = 3071519
    midpoint = int(offs + (smith_starts[FN_ind[count]]+smith_ends[FN_ind[count]])/2.0)
    
    ax4.plot(Cass_data["BTotal(nT)"][midpoint - 60 : midpoint + 60])
    
    ax4.axvspan(Cass_data.index[offs + smith_starts[FN_ind[count]]], 
        Cass_data.index[offs + smith_ends[FN_ind[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
            
    ax4lim = ax4.get_ylim()
    ax4.set_ylabel("$|B|\ (nT)$", fontsize = 20)
    ax4.set_xticklabels([])
    ax4.legend()
    
    ax1.plot(Cass_data["BX_KRTP(nT)"][midpoint - 60 : midpoint + 60])
    ax1lim = ax1.get_ylim()
    ax1.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':',color='k')
    
    ax1.axvspan(Cass_data.index[offs + smith_starts[FN_ind[count]]], 
        Cass_data.index[offs + smith_ends[FN_ind[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
            
    ax1.set_ylabel("$B_r\ (nT)$", fontsize = 20)
    ax1.set_xticklabels([])
    ax1.legend()
    
    ax2.plot(Cass_data["BY_KRTP(nT)"][midpoint - 60 : midpoint + 60])
    ax2lim = ax2.get_ylim()
    ax2.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':', color='k')

    
    ax2.axvspan(Cass_data.index[offs + smith_starts[FN_ind[count]]], 
        Cass_data.index[offs + smith_ends[FN_ind[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
            
    ax2.set_ylabel("$B_\\theta\ (nT)$", fontsize = 20)
    ax2.set_xticklabels([])
    ax2.legend()
    
    ax3.plot(Cass_data["BZ_KRTP(nT)"][midpoint - 60 : midpoint + 60])
    ax3lim = ax3.get_ylim()
    ax3xlim = ax3.get_xlim()[0]
    ax3.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':', color='k')

    ax3.axvspan(Cass_data.index[offs + smith_starts[FN_ind[count]]], 
        Cass_data.index[offs + smith_ends[FN_ind[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
            
    ax3.set_ylabel("$B_\\phi\ (nT)$", fontsize = 20)
    ax3.legend()
    
    ymin=np.min([ax1lim[0],ax2lim[0],ax3lim[0]])
    ymax=np.max([ax1lim[1],ax2lim[1],ax3lim[1]])
    
    ax4.set_ylim([0,ax4lim[1]])
    ax1.set_ylim([ymin,ymax])
    ax2.set_ylim([ymin,ymax])
    ax3.set_ylim([ymin,ymax])
    
    date_form = matplotlib.dates.DateFormatter("%H:%M")
    ax3.xaxis.set_major_formatter(date_form)

    fig.canvas.draw()
    labels = [item.get_text() for item in ax3.get_xticklabels()]
    
    date_form = matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M")
    ax3.xaxis.set_major_formatter(date_form)
    
    ax3.set_ylim([ymin,ymax])

    fig.canvas.draw()
    label_dates = [item.get_text() for item in ax3.get_xticklabels()]
    
    for i in range(len(labels)):
        tick_date = np.where(dat_index == (label_dates[i]+':30'))[0][0]
        labels[i] = (labels[i] + '\n'+str(round(r.iloc[tick_date],2)) + '\n'+str(round(t.iloc[tick_date],2)) + '\n'+str(hours[tick_date])+':'+str(minutes[tick_date].astype(int)).rjust(2,'0'))
    
    ax3.set_xticklabels(labels)
    
    ax4.tick_params(axis='y', which='major', labelsize=14)
    ax1.tick_params(axis='y', which='major', labelsize=14)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    ax3.set_ylim([ymin,ymax])
        
    fig.savefig(dir_path+'\\Images\\FN_Events\\Event_'+str(count)+'.png')
    plt.close(fig)


FP_dbth = []
FP_dur = []
FP_lt = []
FP_rad = []

for count in range(len(FP_ind)):
    FP_dbth = np.append(FP_dbth, np.max(Cass_data["BY_KRTP(nT)"].iloc[starts[FP_ind[count]]+3071519:ends[FP_ind[count]]+3071520]) - np.min(Cass_data["BY_KRTP(nT)"].iloc[starts[FP_ind[count]]+3071519:ends[FP_ind[count]]+3071520]))
    FP_dur = np.append(FP_dur, ends[FP_ind[count]]-starts[FP_ind[count]])
    FP_lt = np.append(FP_lt, lt[starts[FP_ind[count]]+3071519])
    FP_rad = np.append(FP_rad, r[starts[FP_ind[count]]+3071519])

    print(count)
    gridsize=(4,2)
    
    fig = plt.figure(figsize=(20,12))

    ax5 = plt.subplot2grid(gridsize,(0,0), rowspan = 4) 
    
    for i in range(0,len(x_fill),2):
        plt.fill([0,x_fill[i],x_fill[i+1]],[0,y_fill[i],y_fill[i+1]],color="grey", alpha = 0.1)
    for i in range(10,100,10):
        rc = i
        
        xc = rc*np.cos(th_c)
        yc = rc*np.sin(th_c)
        
        plt.plot(xc,yc, color = 'k', linestyle = (0, (1, 10)), linewidth = 1)
    
    plt.scatter(x[starts[FP_ind]+3071518]/rs,y[starts[FP_ind]+3071518]/rs,s=50, color = 'r', 
                label = 'False Negative Events')
    plt.scatter(x[starts[FP_ind[count]]+3071518]/rs,y[starts[FP_ind[count]]+3071518]/rs,s=500, color = 'g', marker = 'X')
    plt.scatter(0,0, color='k', s=500.0, marker = 'o', label = "Saturn")
    plt.ylim([-20,70])
    plt.xlim([-30,30])
    plt.xlabel('X/R$_S$', fontsize = 20)
    plt.ylabel('Y/R$_S$', fontsize = 20)
    plt.title(str(Cass_data.index[starts[FP_ind[count]]+3071518]), fontsize = 20)

    plt.legend(fontsize = 16)
    
    plt.text(31.5,-29,'Hour (UT) \nRad (R$_S$) \nLat ($^\circ$) \nLT (hr:min)', fontsize = 13)
    

    ax4 = plt.subplot2grid(gridsize,(0,1))
    plt.title("Cassini Magnetic Measurements", fontsize = 20)
    ax1 = plt.subplot2grid(gridsize,(1,1))
    ax2 = plt.subplot2grid(gridsize,(2,1))
    ax3 = plt.subplot2grid(gridsize,(3,1))
    
    offs = 3071519
    midpoint = int(offs + (starts[FP_ind[count]]+ends[FP_ind[count]])/2.0)
    
    ax4.plot(Cass_data["BTotal(nT)"][midpoint - 60 : midpoint + 60])
    
    ax4.axvspan(Cass_data.index[offs + starts[FP_ind[count]]], 
        Cass_data.index[offs + ends[FP_ind[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
            
    ax4lim = ax4.get_ylim()
    ax4.set_ylabel("$|B|\ (nT)$", fontsize = 20)
    ax4.set_xticklabels([])
    ax4.legend()
    
    ax1.plot(Cass_data["BX_KRTP(nT)"][midpoint - 60 : midpoint + 60])
    ax1lim = ax1.get_ylim()
    ax1.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':',color='k')
    
    ax1.axvspan(Cass_data.index[offs + starts[FP_ind[count]]], 
        Cass_data.index[offs + ends[FP_ind[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
            
    ax1.set_ylabel("$B_r\ (nT)$", fontsize = 20)
    ax1.set_xticklabels([])
    ax1.legend()
    
    ax2.plot(Cass_data["BY_KRTP(nT)"][midpoint - 60 : midpoint + 60])
    ax2lim = ax2.get_ylim()
    ax2.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':', color='k')

    
    ax2.axvspan(Cass_data.index[offs + starts[FP_ind[count]]], 
        Cass_data.index[offs + ends[FP_ind[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
            
    ax2.set_ylabel("$B_\\theta\ (nT)$", fontsize = 20)
    ax2.set_xticklabels([])
    ax2.legend()
    
    ax3.plot(Cass_data["BZ_KRTP(nT)"][midpoint - 60 : midpoint + 60])
    ax3lim = ax3.get_ylim()
    ax3xlim = ax3.get_xlim()[0]
    ax3.plot(Cass_data.index[midpoint - 60 : midpoint +60], np.zeros(120), linestyle = ':', color='k')

    ax3.axvspan(Cass_data.index[offs + starts[FP_ind[count]]], 
        Cass_data.index[offs + ends[FP_ind[count]]], 
        color = 'red', alpha = 0.5, label = 'ML Event')
            
    ax3.set_ylabel("$B_\\phi\ (nT)$", fontsize = 20)
    ax3.legend()
    
    ymin=np.min([ax1lim[0],ax2lim[0],ax3lim[0]])
    ymax=np.max([ax1lim[1],ax2lim[1],ax3lim[1]])
    
    ax4.set_ylim([0,ax4lim[1]])
    ax1.set_ylim([ymin,ymax])
    ax2.set_ylim([ymin,ymax])
    ax3.set_ylim([ymin,ymax])
    
    date_form = matplotlib.dates.DateFormatter("%H:%M")
    ax3.xaxis.set_major_formatter(date_form)

    fig.canvas.draw()
    labels = [item.get_text() for item in ax3.get_xticklabels()]
    
    date_form = matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M")
    ax3.xaxis.set_major_formatter(date_form)
    
    ax3.set_ylim([ymin,ymax])

    fig.canvas.draw()
    label_dates = [item.get_text() for item in ax3.get_xticklabels()]
    
    for i in range(len(labels)):
        tick_date = np.where(dat_index == (label_dates[i]+':30'))[0][0]
        labels[i] = (labels[i] + '\n'+str(round(r.iloc[tick_date],2)) + '\n'+str(round(t.iloc[tick_date],2)) + '\n'+str(hours[tick_date])+':'+str(minutes[tick_date].astype(int)).rjust(2,'0'))
    
    ax3.set_xticklabels(labels)
    
    ax4.tick_params(axis='y', which='major', labelsize=14)
    ax1.tick_params(axis='y', which='major', labelsize=14)
    ax2.tick_params(axis='y', which='major', labelsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    ax3.set_ylim([ymin,ymax])
        
    fig.savefig(dir_path+'\\Images\\FP_Events\\Event_'+str(count)+'.png')
    plt.close(fig)

gridsize=(3,4)

dur_bins = [0.,2.5,5.,7.5,10.,12.5,15.,17.5,20.,22.5,25.,27.5,30.,32.5,35.,37.5,40.]
dbth_bins = [0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5]
lt_bins = [18.,18.5,19.,19.5,20.,20.5,21.,21.5,22.]
rad_bins =  [16,21,26,31,36,41,46,51]
   
fig = plt.figure(figsize=(30,12))


ax1 = plt.subplot2grid(gridsize,(0,0))
ax1.hist(TP_dur, bins = dur_bins, color = 'green', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax1.set_ylabel('True Positives', fontsize = 20)
ax2 = plt.subplot2grid(gridsize,(1,0))
ax2.hist(FP_dur, bins = dur_bins, color = 'orange', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax2.set_ylabel('False Positives', fontsize = 20)
ax3 = plt.subplot2grid(gridsize,(2,0))
ax3.hist(FN_dur, bins = dur_bins, color = 'red', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax3.set_ylabel('False Negatives', fontsize = 20)
ax3.set_xlabel('Duration (mins)', fontsize = 20)

ax4 = plt.subplot2grid(gridsize,(0,1))
ax4.hist(TP_dbth, bins = dbth_bins, color = 'green', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax5 = plt.subplot2grid(gridsize,(1,1))
ax5.hist(FP_dbth, bins = dbth_bins, color = 'orange', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax6 = plt.subplot2grid(gridsize,(2,1))
ax6.hist(FN_dbth, bins = dbth_bins, color = 'red', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax6.set_xlabel('$\Delta B_\\theta\ (nT)$', fontsize = 20)

ax7 = plt.subplot2grid(gridsize,(0,2))
ax7.hist(TP_lt, bins = lt_bins, color = 'green', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax8 = plt.subplot2grid(gridsize,(1,2))
ax8.hist(FP_lt, bins = lt_bins, color = 'orange', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax9 = plt.subplot2grid(gridsize,(2,2))
ax9.hist(FN_lt, bins = lt_bins, color = 'red', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax9.set_xlabel('Local Time (hrs)', fontsize = 20)

ax10 = plt.subplot2grid(gridsize,(0,3))
ax10.hist(TP_rad, bins = rad_bins, color = 'green', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax11 = plt.subplot2grid(gridsize,(1,3))
ax11.hist(FP_rad, bins = rad_bins, color = 'orange', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax12 = plt.subplot2grid(gridsize,(2,3))
ax12.hist(FN_rad, bins = rad_bins, color = 'red', alpha = 0.5, edgecolor='black', label = "Smith durations")
ax12.set_xlabel('Radial Distance (R$_S$)', fontsize = 20)

ax1.tick_params(axis='both', which='major', labelsize=18)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax3.tick_params(axis='both', which='major', labelsize=18)
ax4.tick_params(axis='both', which='major', labelsize=18)
ax5.tick_params(axis='both', which='major', labelsize=18)
ax6.tick_params(axis='both', which='major', labelsize=18)
ax7.tick_params(axis='both', which='major', labelsize=18)
ax8.tick_params(axis='both', which='major', labelsize=18)
ax9.tick_params(axis='both', which='major', labelsize=18)
ax10.tick_params(axis='both', which='major', labelsize=18)
ax11.tick_params(axis='both', which='major', labelsize=18)
ax12.tick_params(axis='both', which='major', labelsize=18)


TPE=TPE[0]
FPE=FPE[0]
FNE=FNE[0]

ev_list=np.zeros(Cass_data.shape[0])
ev_list[3071518:(3071518+525599)] = ml_events

TNE = np.where((ev_list == 0) &(Cass_data.index >= datetime(2010,1,1)) & (Cass_data.index < datetime(2011,1,1)) & (location == 0) & ((lt > 18) | (lt < 6)) & (r > 15))[0].shape[0]/np.mean(TP_dur)

Ev_HSS = 2.*((TPE*TNE) - (FNE*FPE))/((TPE + FNE)*(FNE + TNE) + (TPE + FPE)*(FPE + TNE)) # Heidke Skill Score
Ev_TSS = (TPE/(TPE + FNE)) - (FPE/(FPE + TNE)) # True Skill Score
Ev_BIAS = (TPE + FPE)/(TPE + FNE) # Bias of Detection
Ev_POD = TPE/(TPE + FNE) # Probability of Detection
Ev_POFD = FPE/(TNE + FPE) # Probablity of False Detection
Ev_FAR = FPE/(TPE + FPE) # False Alarm Ratio
Ev_TS = TPE/(TPE + FNE + FPE) # Threat Score
Ev_OR = (TPE*TNE)/(FNE*FPE) # Odds Ratio

"""

#=============#
#=====EOF=====#
#=============#