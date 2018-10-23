# import tensorflow as tf
import numpy as np
import pandas as pd
# import cv2
import os
from tqdm import tqdm
import shutil
# from keras import models, layers

def get_name(path):
    path_split = path.split('.')
    return path_split[0]

def get_class(path):
    name = get_name(path)
    if name == "cat":
        return np.array([1, 0])
    else:
        return np.array([0, 1])

def main(test_size, validation_size):
    base_dir = os.path.join(os.getcwd(), "data")
    
    train_dir = os.path.join(base_dir, "train")
    try:
        shutil.rmtree(train_dir)
    except FileNotFoundError:
        print("Creating a Train folder")

    os.mkdir(train_dir)
    os.mkdir(os.path.join(train_dir, "cats"))
    os.mkdir(os.path.join(train_dir, "dogs"))
    validation_dir = os.path.join(base_dir, "validation")
    try:
        shutil.rmtree(validation_dir)
    except FileNotFoundError:
        print("Creating new Validation folder")
    os.mkdir(validation_dir)
    os.mkdir(os.path.join(validation_dir, "cats"))
    os.mkdir(os.path.join(validation_dir, "dogs"))

    test_dir = os.path.join(base_dir, "test")
    try:
        shutil.rmtree(test_dir)
    except FileNotFoundError:
        print("Creating Test folder")

    os.mkdir(test_dir)
    os.mkdir(os.path.join(test_dir, "cats"))
    os.mkdir(os.path.join(test_dir, "dogs"))
    
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

    # split train valid test
    # images 0 to num_test are used in the test set
    for i in tqdm(range(num_test_imgs // 2), desc="Building test set for Cats"):
        shutil.move(os.path.join(base_path, cats[i]), os.path.join(test_dir, "cats"))
    
    for i in tqdm(range(num_test_imgs // 2, 0, -1), desc="Building test set Dogs"):
        shutil.move(os.path.join(base_path, dogs[i]), os.path.join(test_dir, "dogs"))
    
    # from num_test to num_test + val_size
    validation_dir_size = int((image_set_size - num_test_imgs) * validation_size)
    for i in tqdm(range(num_test_imgs, num_test_imgs + validation_dir_size // 2), desc="Building validation set Cats"):
        shutil.move(os.path.join(base_path, cats[i]), os.path.join(validation_dir, "cats"))
    
    for i in tqdm(range(num_test_imgs + validation_dir_size // 2, num_test_imgs, -1), desc="Building validation set for Dogs"):
        shutil.move(os.path.join(base_path, dogs[i]), os.path.join(validation_dir, "dogs"))
    
    # The rest of the images are used for training
    train_size = image_set_size - validation_dir_size - num_test_imgs
    for i in tqdm(range(num_test_imgs, validation_dir_size + train_size // 2), desc="Building train set Cats"):
        try:
            shutil.move(os.path.join(base_path, cats[i]), os.path.join(train_dir, "cats"))
        except FileNotFoundError as e:
            pass
        

    for i in tqdm(range(train_size // 2 + validation_dir_size, num_test_imgs, -1), desc="Building train set Dogs"):
        try:
            shutil.move(os.path.join(base_path, dogs[i]), os.path.join(train_dir, "dogs"))
        except FileNotFoundError as e:
            pass
    


    # process data

    # build model
    
    # model = models.Sequential()
    # model.add(layers.Conv2D)
    
    # train model

    # test model

    # use model

if __name__ == '__main__':
    main(0.2, 0.2)