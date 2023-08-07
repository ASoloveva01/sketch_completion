import zipfile
from pathlib import Path
import shutil
import os
curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, "data")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

def customize_dataset(num_classes=250, train_size=0.8):



    os.mkdir(train_dir)
    os.mkdir(test_dir)

    data_path = Path("data/")
    images_path = data_path / "sketches_png"
    with zipfile.ZipFile(data_path / "sketches_png.zip", "r") as zip_ref:\
       zip_ref.extractall(images_path)

    images_path_classes_list = list(data_path.glob("*/*/*/"))

    for class_path in images_path_classes_list[:num_classes]:
        class_image_names = os.listdir(class_path)
        train_class_path = os.path.join(train_dir, class_path.stem)
        test_class_path = os.path.join(test_dir, class_path.stem)
        threshold = round(train_size * len(class_image_names))

        os.mkdir(train_class_path)
        os.mkdir(test_class_path)

        for train_image_name in class_image_names[:threshold]:
            src_path = os.path.join(str(class_path), train_image_name)
            shutil.move(src_path, train_class_path)

        for test_image_name in class_image_names[threshold:]:
            src_path = os.path.join(str(class_path), test_image_name)
            shutil.move(src_path, test_class_path)

    shutil.rmtree(images_path)