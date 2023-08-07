import os
import pathlib
from PIL import Image
from torch.utils.data.dataset import Dataset
from typing import Tuple, Dict, List

class CreateDataset(Dataset):

    """Инициализируем целевую директорию и преобразования для замаскированных
    и исходных изображений соответственно"""

    def __init__(self, targ_dir: str, mask_transform, real_transform):

        self.paths = list(
            pathlib.Path(targ_dir).glob("*/*.png"))

        self.mask_transform = mask_transform
        self.real_transform = real_transform

        self.classes, self.class_to_idx = find_classes(targ_dir)
    def load_image(self, index):
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        """Возвращает преобразованное замаскированное изображение, преобразованное исходное изображение
        и индекс класса"""
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        return self.mask_transform(img),  self.real_transform(img), class_idx
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Возвращает кортеж из списка названий класса и словаря(имя класса: индекс)"""

    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Невозможно найти классы в {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
