import os
import glob
import natsort
import interface_file_io as file_io


def get_directory_to_label(dataset_dir):
    data_list = natsort.natsorted(os.listdir(dataset_dir))
    for idx, data in enumerate(data_list):
        data = data.lower()
        data_list[idx] = data.replace(' ', '_')
    return data_list


def get_label(dataset_dir_list, output_filepath):
    data_list = []
    for dataset_dir in dataset_dir_list:
        data_list += get_directory_to_label(dataset_dir)
    data_list = natsort.natsorted(list(set(data_list)))
    file_io.make_list2txt(data_list, output_filepath)


def get_label(dataset_dir, output_file):
    label_list = os.listdir(dataset_dir)
    label_list = list(set(label_list))
    file_io.make_list2txt(label_list, output_file)


if __name__ == '__main__':
    dir_list = [
        '../../dataset/FruitsandVegs/train',
        '../../dataset/VegetableImages/train'
    ]

    output = '../../dataset/FV-label.txt'
    get_label(dir_list, output)