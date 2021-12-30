import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from src.utils import interface_file_io as file_io
import torchvision.io as io
import torchvision.transforms as transforms


def remove_mac_keyword(data):
    if '.DS_Store' in data:
        data.remove('.DS_Store')
    return data


# num_classes = 30
class KFoodSampleDataset(Dataset):
    def __init__(self, dataset_path):
        super(KFoodSampleDataset, self).__init__()
        self.label_list = remove_mac_keyword(sorted(os.listdir(dataset_path)))
        self.file_list = file_io.get_all_file_path(dataset_path, 'jpg')
        self.image_transforms = nn.Sequential(
            transforms.RandomResizedCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
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




