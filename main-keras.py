import horovod.tensorflow.keras as hvd
import numpy as np 
import h5py
import tensorflow as tf
from lib import cnn

# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def loadDataH5(): 
    with h5py.File('./files/data1.h5','r') as hf: 
        trainX = np.array(hf.get('trainX')) 
        trainY = np.array(hf.get('trainY')) 
        valX = np.array(hf.get('valX')) 
        valY = np.array(hf.get('valY')) 
        print (trainX.shape,trainY.shape) 
        print (valX.shape,valY.shape) 
    return trainX, trainY, valX, valY 

if __name__ == '__main__':
    trainX, trainY, testX, testY = loadDataH5()
    
    NUM_EPOCHS =50
    inshape=trainX.shape[1:]
    classes=np.unique(trainY).size
    
    # Horovod: adjust learning rate based on lr_scaler.
    opt = tf.keras.optimizers.SGD(lr=0.01 * hvd.size())

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
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint.h5'))


    print (model.summary())
    print("Training network...")
    History = model.fit(trainX, trainY, batch_size=16, 
                epochs=NUM_EPOCHS, validation_data=(testX, testY), callbacks=callbacks, verbose=1 if hvd.rank() == 0 else 0)
    print ("Test Data Loss and Accuracy: ", model.evaluate(testX, testY))