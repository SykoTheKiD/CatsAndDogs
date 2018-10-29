import os
from tqdm import tqdm
import shutil
import numpy as np


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
