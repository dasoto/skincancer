import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from moleimages import MoleImages


# dimensions of our images.
img_width, img_height = 128, 128

top_model_weights_path = 'models/bottleneck_fc_model.h5'
train_data_dir = 'data_scaled'
validation_data_dir = 'data_scaled_validation'
nb_train_samples = 1760 #1763
nb_validation_samples = 192 #194
epochs = 50
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator()  #rescale = 1. /255

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('models/bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('models/bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('models/bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (1043) + [1] * (717))

    validation_data = np.load(open('models/bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (115) + [1] * (77))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adadelta',  #rmsprop
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    print('saving weights file: ',top_model_weights_path)
    return model

if __name__ == '__main__':
    save_bottlebeck_features()
    model = train_top_model()
