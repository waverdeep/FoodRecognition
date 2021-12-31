import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from src.utils import interface_file_io as file_io
import torchvision
import torchvision.io as io
import torchvision.transforms as transforms
import PIL


# num_classes = 30
class KFoodSampleDataset(Dataset):
    def __init__(self, directory_path):
        super(KFoodSampleDataset, self).__init__()
        self.label_list = sorted(os.listdir(directory_path))
        self.file_list = file_io.get_all_file_path(directory_path, 'jpg')
        self.image_transforms = nn.Sequential(
            transforms.RandomResizedCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, x):
        # './dataset/sample_food_data/갈비구이/Img_000_0000.jpg'
        file = self.file_list[x]
        label = self.label_list.index(file.split('/')[3])
        data = io.read_image(file)
        data = data.type('torch.FloatTensor')
        data = self.image_transforms(data)
        return data, label


# num_classes = 150 (detail)
# num_classes = 27
class KFoodDataset(Dataset):
    def __init__(self, directory_path, mode='train'):
        super(KFoodDataset, self).__init__()
        self.detail_label_list = file_io.read_txt2list('./dataset/kfood-category-detail.txt')
        self.label_list = file_io.read_txt2list('./dataset/kfood-category.txt')
        self.file_list = file_io.read_txt2list(directory_path)
        if mode == 'train':
            self.image_transforms = nn.Sequential(
                transforms.ToTensor(),
                transforms.RandomResizedCrop(512),
                # transforms.RandomHorizontalFlip(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # transforms.RandomRotation(degrees=(0, 360)),

            )
        else:
            self.image_transforms = nn.Sequential(
                # transforms.Resize(640),
                transforms.CenterCrop(512),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, x):
        # './dataset/kfood/구이/갈비구이/Img_000_0000.jpg'
        file = self.file_list[x]
        print(file)
        label = self.label_list.index(file.split('/')[3])
        detail_label = self.detail_label_list.index(file.split('/')[4])
        # have issue - png 4channel cannot load only supported 3 dimension
        data = PIL.Image.open(file)
        # data = data.type('torch.FloatTensor')
        data = self.image_transforms(data)
        return data, label, detail_label


