import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from src.utils import interface_file_io as file_io
import torchvision
import torchvision.io as io
import torchvision.transforms as transforms
import PIL


# num_classes = 36
class IngredientDataset(Dataset):
    def __init__(self, directory_path, mode='train', crop_size=512):
        super(IngredientDataset, self).__init__()
        self.label_list = file_io.read_txt2list('./dataset/FruitandVegsLabel.txt')
        self.file_list = file_io.read_txt2list(directory_path)
        if mode == 'train':
            self.image_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=(0, 360)),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(640),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, x):
        # './dataset/FruitandVegs/train/apple/Img_000_0000.jpg'
        file = self.file_list[x]
        label = self.label_list.index(file.split('/')[4])
        # have issue - png 4channel cannot load only supported 3 dimension
        data = PIL.Image.open(file)
        # data = data.type('torch.FloatTensor')
        data = self.image_transforms(data)
        return data, label, label
