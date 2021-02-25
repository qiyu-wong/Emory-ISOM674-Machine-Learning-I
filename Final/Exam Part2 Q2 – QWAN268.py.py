# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:03:29 2018

@author: GEASTON
"""
#%% import packages, set some parameters, and get the data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

NEpochs = 10000
BatchSize=250
Optimizer=optimizers.RMSprop(lr=0.001)

# Read in the data

TrainDF = pd.read_csv('C:/Users/10331/OneDrive/Desktop/TrainDF.csv',sep=',',header=0,quotechar='"')
list(TrainDF)
ValDF = pd.read_csv('C:/Users/10331/OneDrive/Desktop/ValDF.csv',sep=',',header=0,quotechar='"')
list(ValDF)

#

TrY = np.array(TrainDF['Y'])
TrX = np.array(TrainDF.iloc[:,1:])

TrXrsc = (TrX - TrX.min(axis=0))/TrX.ptp(axis=0)
print(TrXrsc.shape)
print(TrXrsc.min(axis=0))
print(TrXrsc.max(axis=0))

# No need to rescale the Y because it is already 0 and 1. But check
print(TrY.min())
print(TrY.max())

# Rescale the validation data

ValY = np.array(ValDF['Y'])

ValX = np.array(ValDF.iloc[:,1:])

ValXrsc = (ValX - TrX.min(axis=0))/TrX.ptp(axis=0)
print(ValXrsc.shape)
print(ValXrsc.min(axis=0))
print(ValXrsc.max(axis=0))

print(ValY.min())
print(ValY.max())


#%% Set up test Neural Net Model

SpamNN = Sequential()

SpamNN.add(Dense(units=5,input_shape=(TrXrsc.shape[1],),activation="relu",use_bias=True))
SpamNN.add(Dense(units=5,activation="relu",use_bias=True))
SpamNN.add(Dense(units=1,activation="sigmoid",use_bias=True))

SpamNN.compile(loss='binary_crossentropy', optimizer=Optimizer,metrics=['binary_crossentropy','accuracy'])
print(SpamNN.summary())
#%% Fit NN Model

from keras.callbacks import EarlyStopping

StopRule = EarlyStopping(monitor='val_loss',mode='min',verbose=0,patience=100,min_delta=0.0)
FitHist = SpamNN.fit(TrXrsc,TrY,validation_data=(ValXrsc,ValY), \
                    epochs=NEpochs,batch_size=BatchSize,verbose=0, \
                    callbacks=[StopRule])
    
#FitHist = SpiralNN.fit(TrXrsc,TrColorCode,epochs=NEpochs,batch_size=BatchSize,verbose=0)

print("Number of Epochs = "+str(len(FitHist.history['accuracy'])))
print("Final training accuracy: "+str(FitHist.history['accuracy'][-1]))
print("Recent history for training accuracy: "+str(FitHist.history['accuracy'][-10:-1]))
print("Final validation accuracy: "+str(FitHist.history['val_accuracy'][-1]))
print("Recent history for validation accuracy: "+str(FitHist.history['val_accuracy'][-10:-1]))

#%% Make Predictions

TrP = SpamNN.predict(TrXrsc,batch_size=TrXrsc.shape[0])
ValP = SpamNN.predict(ValXrsc,batch_size=TrXrsc.shape[0])

#%% Write out prediction

TrainDF['TrP'] = TrP.reshape(-1)
ValDF['ValP'] = ValP.reshape(-1)

TrainDF.to_csv('C:/Users/10331/OneDrive/Desktop/TrainDFOutput.csv',sep=',',na_rep="NA",header=True,index=False)
ValDF.to_csv('C:/Users/10331/OneDrive/Desktop/ValDFOutput.csv',sep=',',na_rep="NA",header=True,index=False)


#%% Set up narrow but deep Neural Net Model

SpamNN = Sequential()

SpamNN.add(Dense(units=4,input_shape=(TrXrsc.shape[1],),activation="relu",use_bias=True))
SpamNN.add(Dense(units=4,activation="relu",use_bias=True))
SpamNN.add(Dense(units=4,activation="relu",use_bias=True))
SpamNN.add(Dense(units=4,activation="relu",use_bias=True))
SpamNN.add(Dense(units=4,activation="relu",use_bias=True))
SpamNN.add(Dense(units=1,activation="sigmoid",use_bias=True))

SpamNN.compile(loss='binary_crossentropy', optimizer=Optimizer,metrics=['binary_crossentropy','accuracy'])
print(SpamNN.summary())
#%% Fit NN Model

from keras.callbacks import EarlyStopping

StopRule = EarlyStopping(monitor='val_loss',mode='min',verbose=0,patience=100,min_delta=0.0)
FitHist = SpamNN.fit(TrXrsc,TrY,validation_data=(ValXrsc,ValY), \
                    epochs=NEpochs,batch_size=BatchSize,verbose=0, \
                    callbacks=[StopRule])
    
#FitHist = SpiralNN.fit(TrXrsc,TrColorCode,epochs=NEpochs,batch_size=BatchSize,verbose=0)

print("Number of Epochs = "+str(len(FitHist.history['accuracy'])))
print("Final training accuracy: "+str(FitHist.history['accuracy'][-1]))
print("Recent history for training accuracy: "+str(FitHist.history['accuracy'][-10:-1]))
print("Final validation accuracy: "+str(FitHist.history['val_accuracy'][-1]))
print("Recent history for validation accuracy: "+str(FitHist.history['val_accuracy'][-10:-1]))

#%% Make Predictions

TrP = SpamNN.predict(TrXrsc,batch_size=TrXrsc.shape[0])
ValP = SpamNN.predict(ValXrsc,batch_size=TrXrsc.shape[0])

#%% Write out prediction

TrainDF['TrP'] = TrP.reshape(-1)
ValDF['ValP'] = ValP.reshape(-1)

TrainDF.to_csv('C:/Users/10331/OneDrive/Desktop/TrainDFOutput2.csv',sep=',',na_rep="NA",header=True,index=False)
ValDF.to_csv('C:/Users/10331/OneDrive/Desktop/ValDFOutput2.csv',sep=',',na_rep="NA",header=True,index=False)