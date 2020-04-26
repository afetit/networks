# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import config
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import os

NUM_EPOCHS = 100
#INIT_LR = 0.0001
BS = 32
save_dir = "Data/cifar/saved_models"
model_name = 'keras_cifar10_trained_model.h5'

totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

trainAug = ImageDataGenerator(rescale=1)
valAug = ImageDataGenerator(rescale=1)

trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    color_mode="rgb",
    target_size=(32,32),
    shuffle=True,
    batch_size=BS)

valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    color_mode="rgb",
    target_size=(32,32),
    shuffle=False,
    batch_size=BS)

testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    color_mode="rgb",
    target_size=(32,32),
    shuffle=False,
    batch_size=BS)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen,
    validation_steps=totalVal // BS,
    epochs=NUM_EPOCHS)

# reset the testing generator and then use our trained model to make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen, steps=(totalTest // BS) + 1)

# find the index of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print(f'Saved trained model at {model_path}')