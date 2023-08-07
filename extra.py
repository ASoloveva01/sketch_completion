import torch
import zipfile
from pathlib import Path
import shutil
import os
def load_dataset():
    curr_dir = os.getcwd()

#data_dir = curr_dir + "/data"
#os.mkdir(data_dir+"/train")
#os.mkdir(data_dir+"/test")
    data_dir = os.path.join(curr_dir, "data")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    data_path = Path("data/")
    images_path = data_path / "sketches_png"
    with zipfile.ZipFile(data_path / "sketches_png.zip", "r") as zip_ref:\
       zip_ref.extractall(images_path)
    images_path_classes_list = list(data_path.glob("*/*/*/"))
    train_perc = 0.95

    for class_path in images_path_classes_list[:6]:
        images_per_class_list = os.listdir(class_path)
        train_class_path = os.path.join(train_dir, class_path.stem)
        test_class_path = os.path.join(test_dir, class_path.stem)
        threshold = round(train_perc * len(images_per_class_list))

        os.mkdir(train_class_path)
        os.mkdir(test_class_path)
        for image_name in images_per_class_list[:threshold]:
            src_path = os.path.join(str(class_path), image_name)
            shutil.move(src_path, train_class_path)
        for image_name in images_per_class_list[threshold:]:
            src_path = os.path.join(str(class_path), image_name)
            shutil.move(src_path, test_class_path)
    shutil.rmtree(images_path)