from keras.preprocessing.image import ImageDataGenerator
from cnn_model import CNN

if __name__ == '__main__':
    train_data_dir = 'data_scaled/'
    validation_data_dir = 'data_scaled_validation/'
    nb_train_samples = 1445
    nb_validation_samples = 70
    epochs = 100
    batch_size = 16

    mycnn = CNN()
    train_datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True)
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary')

    model = mycnn.fit_generator(train_generator,validation_generator,
        nb_train_samples, nb_validation_samples, epochs, batch_size)

    model.save('mymodel.h5')
