import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from src.utils import interface_file_io as file_io
import torchvision
import torchvision.io as io
import torchvision.transforms as transforms
import PIL


# num_classes = 256
class INGDDataset(Dataset):
    def __init__(self, directory_path, mode='train', crop_size=512):
        super(INGDDataset, self).__init__()
        self.label_list = file_io.read_txt2list('./dataset/INGD_V2.txt')
        self.file_list = file_io.read_txt2list(directory_path)
        if mode == 'train':
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(640),
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=(0, 360)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

                ]
            )
        else:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(640),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, x):
        # './dataset/INGD_V1/corn/194.jpg'
        file = self.file_list[x]
        label = file.split('/')[3]
        label = label.lower()
        label = label.replace(' ', "_")
        label = self.label_list.index(label)
        # have issue - png 4channel cannot load only supported 3 dimension
        data = PIL.Image.open(file)
        # data = data.type('torch.FloatTensor')
        data = self.image_transforms(data)
        return data, label
