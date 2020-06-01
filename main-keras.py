import horovod.tensorflow.keras as hvd
import numpy as np 
import h5py
import tensorflow as tf
import tensorflow.keras.backend as K
from lib import cnn
from time import time
import math
from datetime import datetime

def print_datetime(msg):
    now = datetime.now()
    print(msg, now)

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

def loadDataH5(): 
    with h5py.File('./files/data1.h5','r') as hf: 
        trainX = np.array(hf.get('trainX')) 
        trainY = np.array(hf.get('trainY')) 
        valX = np.array(hf.get('valX')) 
        valY = np.array(hf.get('valY')) 
        print (trainX.shape,trainY.shape) 
        print (valX.shape,valY.shape) 
    return trainX, trainY, valX, valY 

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.logs=[]
  
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime=time()
  
    def on_epoch_end(self, epoch, logs={}):
        time_interval = time()-self.starttime
        print("Time taken for epoch {} : {}".format(epoch, time_interval))
        self.logs.append(time_interval)

if __name__ == '__main__':
    trainX, trainY, testX, testY = loadDataH5()
    
    NUM_EPOCHS = 100

    # Horovod: adjust number of epochs based on number of Processing units.
    epochs = int(math.ceil(NUM_EPOCHS / hvd.size()))
    batch_size = 16

    inshape=trainX.shape[1:]
    classes=np.unique(trainY).size
    
    # Horovod: adjust learning rate based on lr_scaler.
    opt = tf.keras.optimizers.Adam(lr=0.1 * hvd.size())

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(opt)

    model = cnn.cnmodel.model(inshape,classes)

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        
        TimingCallback()
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint.h5'))

    print_datetime("Training initiated at: ")
    
    print (model.summary())
    print("Training network...")
    History = model.fit(
        trainX, 
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(testX, testY), 
        callbacks=callbacks, 
        verbose=1 if hvd.rank() == 0 else 0)

    print ("Test Data Loss and Accuracy: ", model.evaluate(testX, testY))
    
    print_datetime("Training terminated at: ")