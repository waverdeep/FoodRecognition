import os
import glob
import natsort
import interface_file_io as file_io


def get_directory_to_label(root_dir):
    dir_list = natsort.natsorted(os.listdir(root_dir))
    file_io.make_list2txt(dir_list, '../../dataset/FruitandVegsLabel.txt')


if __name__ == '__main__':
    get_directory_to_label('../../dataset/FruitandVegs/train')