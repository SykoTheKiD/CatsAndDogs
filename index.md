
## Using Deep Learning to Classify Images of Cats and Dogs

### Requirements:
* TensorFlow 1.6
* Keras 2.2.1
* Python 3.6.6


### Getting our data

For this tutorial we will use the popular [Cats and Dogs dataset on Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data). We will download and extract the training images to a `data/imgs/`. We will split these images into train and test sets.

This data set contains *25,000* images in total (12,500 images of Dogs and 12,500 images of Cats).


First our imports:


```python
import os
from tqdm import tqdm
import shutil
import numpy as np
```

The first function will take a image file and return the type of animal in the image. Since our images are of the form animal.number.jpg we can get the first part of the file name to sort our images.

Prepare folders will create two separate directories `train` and `test` that will hold a portions of out dataset to train our model.

Out test dataset is more accurately called a **validation set** since we have a seperate set of images we will use to test our model. And our **test set** is actually a portion of out training set. But to keep it simple we will just refer to is as our test set.

**Repopulate** is just a flag to set whether or not you want to re-copy images from the `data/imgs/` folder again.



```python
def get_name(path):
    fname = os.path.basename(path)
    fname = path.split('.')
    return fname[0]

def prepare_folders(test_size, repopulate=False):
    base_dir = os.path.abspath("data")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    if repopulate:
        try:
            shutil.rmtree(train_dir)
        except FileNotFoundError:
            print("Creating a Train folder")

        os.mkdir(train_dir)
        os.mkdir(os.path.join(train_dir, "cats"))
        os.mkdir(os.path.join(train_dir, "dogs"))

        try:
            shutil.rmtree(test_dir)
        except FileNotFoundError:
            print("Creating Test folder")

        os.mkdir(test_dir)
        os.mkdir(os.path.join(test_dir, "cats"))
        os.mkdir(os.path.join(test_dir, "dogs"))
```

Now that we will have three folders in our `data/` folder. `imgs/`, `train/` and `test/`. Next we will load all the image paths to separate lists based on the image type.


```python
    # load data
    cats = []
    dogs = []
    base_path = os.path.join(os.getcwd(), "data", "imgs")
    for item in tqdm(os.listdir(base_path), desc="Getting path names"):
        name = get_name(item)
        if name == "cat":
            cats.append(item)
        else:
            dogs.append(item)

    assert len(cats) == len(dogs)

    image_set_size = len(os.listdir(base_path))
    num_test_imgs = int(image_set_size * test_size)
```

Now we copy images from our dataset to its respective `train/` or `test/` folder. When splitting up our images we need to make sure that we have around the same number of images for each type in each set. If we train the model with more dog images and less cats images then the model does not have enough information to model the cats and will fail in our test set.


```python
# split train valid test
    if repopulate:
        for i in tqdm(range(num_test_imgs // 2), desc="Building test set for Cats"):
            shutil.copy(os.path.join(
                base_path, cats[i]), os.path.join(test_dir, "cats"))

        for i in tqdm(range(num_test_imgs // 2), desc="Building test set Dogs"):
            shutil.copy(os.path.join(
                base_path, dogs[i]), os.path.join(test_dir, "dogs"))

        for i in tqdm(range(num_test_imgs // 2, len(cats)), desc="Building train set for Cats"):
            shutil.copy(os.path.join(
                base_path, cats[i]), os.path.join(train_dir, "cats"))

        for i in tqdm(range(num_test_imgs // 2, len(dogs)), desc="Building train set Dogs"):
            shutil.copy(os.path.join(
                base_path, dogs[i]), os.path.join(train_dir, "dogs"))
    try:
        train_size_calc = len(os.listdir(os.path.join(
            train_dir, "cats"))) + len(os.listdir(os.path.join(train_dir, "dogs")))
        test_size_calc = len(os.listdir(os.path.join(
            test_dir, "cats"))) + len(os.listdir(os.path.join(test_dir, "dogs")))
        print(
            f"Dataset Size: {image_set_size}\nTrain Size:{train_size_calc}\nTest Size: {test_size_calc}")
    except FileNotFoundError:
        print("No test sets found")

    return base_dir, train_dir, test_dir

base_dir, train_dir, test_dir = prepare_folders(test_size=0.2)
```

Now we have a train set and test set that is split from the orginal dataset. Now we can construct our model.

We will first create and train our own model from scratch.

## Creating our own Convolutional Neural Network

This tutorial does not go into depth on what a CNN is or how it works.

But ultimately a CNN distills an image into a different representation that is easier to train on a artificial neural network and results in the less weights during training.

#### Image Augmentation

Since out dataset is small and to prevent overfitting we will use Image Augmentation to add noise to our training images. Keras provides a well documented easy to use API to do so.


```python
from time import time
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

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
```

### Defining our Model


```python
model = models.Sequential()
model.add(layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
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

model.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc"])
```

### Tracking our Progress with TensorBoard


```python
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
```

### Saving our model


```python
model_save_path = os.path.join(base_dir, "saves")
try:
    os.mkdir(model_save_path)
except FileExistsError:
    print("Previous Models Found")

try:
    model.save(os.path.join(model_save_path,'cats_and_dogs_{}.h5'.format(time())))
except ModuleNotFoundError as e:
    print("No h5py found", e)
except OSError as e:
    print("Write permission denied", e)
```

### Plotting Our Results


```python
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
```

## Using a Pretrained Model -- VGG16

Creating our own model from scratch and training 25,000 images will only get us so far before our results plateau.

We can use **Transfer Learning** which leverages models previously trained for a different use to be adapted to our use case. A popular model to use is the VGG16 model which has 16 layers and can output 1000 different classes.

### Separable Conv 2D

A newer type of Convolutional Layer called the Separable Convolutional Layer is gaining ground and requires less weights. And in Deep Learning if you can get lower weights it is always the better option.


```python
model = models.Sequential()
model.add(layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.SeparableConv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.SeparableConv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer=optimizers.RMSprop(lr=1e-4), metrics=["acc"])
```
