import tensorflow as tf

class cnmodel:
  @staticmethod
  def model(inshape,classes):
    model=tf.keras.models.Sequential()
    #add convolution layer
    model.add(tf.keras.layers.Conv2D(64,(5,5), padding="same",input_shape=inshape,activation='relu'))
    #add Max Pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #add convolution layer
    model.add(tf.keras.layers.Conv2D(128,(5,5), padding="same",activation='relu'))
    #add Max Pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #add convolution layer
    model.add(tf.keras.layers.Conv2D(256,(5,5), padding="same",activation='relu'))
    #add Max Pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #add convolution layer
    model.add(tf.keras.layers.Conv2D(512,(5,5), padding="same",activation='relu'))
    #add Max Pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #Flatten the data
    model.add(tf.keras.layers.Flatten())
    #add Dense layer
    model.add(tf.keras.layers.Dense(2048, activation='relu'))
    #add Final Softmax layer
    model.add(tf.keras.layers.Dense(classes, activation='softmax'))

    return model