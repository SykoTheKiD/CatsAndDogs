import dataset as dt
import os
from time import time
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard


def run(test_size):
    base_dir, train_dir, test_dir = dt.prepare_folders(test_size)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_gen = val_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    model = models.Sequential()
    model.add(layers.SeparableConv2D(
        32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.SeparableConv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.SeparableConv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(loss="binary_crossentropy",
                  optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc"])
    try:
        os.mkdir("logs")
    except FileExistsError:
        print("Previous Logs found")

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_gen,
        validation_steps=50,
        callbacks=[tensorboard]
    )

    model_save_path = os.path.join(base_dir, "saves")
    try:
        os.mkdir(model_save_path)
    except FileExistsError:
        print("Previous Models Found")

    try:
        model.save(os.path.join(model_save_path,
                                'cats_and_dogs_{}.h5'.format(time())))
    except ModuleNotFoundError as e:
        print("No h5py found", e)
    except OSError as e:
        print("Write permission denied", e)

    acc = history.history["acc"]
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation_acc')
    plt.title('Training and Validation accuracy')

    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def main():
    run(0.2)


if __name__ == '__main__':
    main()
