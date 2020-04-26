from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model
from keras.datasets import cifar10
from skimage import feature, color
import numpy as np
import os
from resnet_cifar10_keras import lr_schedule, resnet_layer, resnet_v1

batch_size = 32
epochs = 200
data_augmentation = True
num_classes = 10

subtract_pixel_mean = True
calculate_lbp_maps = True

n = 9
version = 1
depth = n * 6 + 2
model_type = 'ResNet%dv%d' % (depth, version)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]
train_samples = x_train.shape[0]
test_samples = x_test.shape[0]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
steps_per_epoch = train_samples / batch_size

if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean


if calculate_lbp_maps:
    for a in range(train_samples):
        x_train_grey = color.rgb2gray(x_train[a])
        x_train[a][:,:,0] = feature.local_binary_pattern(x_train_grey, 8, 1, method='uniform')
        x_train[a][:,:,1] = feature.local_binary_pattern(x_train_grey, 8, 5, method='uniform')
        x_train[a][:,:,2] = feature.local_binary_pattern(x_train_grey, 8, 10, method='uniform')
    for b in range(test_samples):
        x_test_grey = color.rgb2gray(x_test[b])
        x_test[b][:,:,0] = feature.local_binary_pattern(x_test_grey, 8, 1, method='uniform')
        x_test[b][:,:,1] = feature.local_binary_pattern(x_test_grey, 8, 5, method='uniform')
        x_test[b][:,:,2] = feature.local_binary_pattern(x_test_grey, 8, 10, method='uniform')


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



model = resnet_v1(input_shape=input_shape, depth=depth)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])
model.summary()
print(model_type)

save_dir = "Data/cifar/saved_models/checkpoints_cifar_keras_resnet_example/cifar10_with_lbp_v2_augment"
model_name = 'keras_cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer, lr_scheduler]

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)

else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=False, # divide inputs by std of dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        zca_epsilon=1e-06, # epsilon for ZCA whitening
        rotation_range=0, # randomly rotate images in the range (deg 0 to 180)
        width_shift_range=0.1, # randomly shift images horizontally
        height_shift_range=0.1, # randomly shift images vertically
        shear_range=0., # set range for random shear
        zoom_range=0., # set range for random zoom
        channel_shift_range=0., # set range for random channel shifts
        fill_mode='nearest', # set mode for filling points outside the input boundaries
        cval=0., # value used for fill_mode = "constant"
        horizontal_flip=True, # randomly flip images
        vertical_flip=False, # randomly flip images
        rescale=None, # set rescaling factor (applied before any other transformation)
        preprocessing_function=None, # set function that will be applied on each input
        data_format=None, # image data format, either "channels_first" or "channels_last"
        validation_split=0.0) # fraction of images reserved for validation (strictly between 0 and 1)

    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs,
                        verbose=1,
                        workers=4,
                        callbacks=callbacks,
                        steps_per_epoch=steps_per_epoch)


scores = model.evaluate(x_test, y_test, verbose=1)
print(f'Test loss: {scores[0]}')
print(f'Test accuracy: {scores[1]}')