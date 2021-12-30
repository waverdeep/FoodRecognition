import glob
import os
import json
import csv


def read_csv_file(filename):
    dataset = []
    with open(filename, 'r', encoding='utf-8') as file:
        data = csv.reader(file)
        for line in data:
            dataset.append(line)
    return dataset


def get_pure_filename(filename):
    temp = filename.split('.')
    del temp[-1]
    temp = '.'.join(temp)
    temp = temp.split('/')
    temp = temp[-1]
    return temp


def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp


def load_json_config(filename):
    with open(filename, 'r') as configuration:
        config = json.load(configuration)
    return config


def make_directory(directory_name, format_logger=None):
    try:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
    except OSError:
        if format_logger is not None:
            format_logger.info('Error: make directory: {}'.format(directory_name))
        else:
            print('Error: make directory: {}'.format(directory_name))


def read_txt2list(file_path):
    with open(file_path, 'r') as data:
        file_list = [x.strip() for x in data.readlines()]
    return file_list


def make_list2txt(file_list, file_path):
    with open('{}'.format(file_path), 'w') as output_file:
        for index, file in enumerate(file_list):
            output_file.write("{}\n".format(file))


def list_divider(step, data):
    split_len = int(len(data)/step)
    return [data[i:i+split_len] for i in range(0, len(data), split_len)]
