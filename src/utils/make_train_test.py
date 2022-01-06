import src.utils.interface_file_io as file_io
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


def convert_jpg(directory_path):
    file_list = []
    file_list += file_io.get_all_file_path(directory_path, 'jpg')
    file_list += file_io.get_all_file_path(directory_path, 'JPG')
    file_list += file_io.get_all_file_path(directory_path, 'png')
    file_list += file_io.get_all_file_path(directory_path, 'jpeg')
    file_list += file_io.get_all_file_path(directory_path, 'bmp')

    for file in tqdm(file_list, desc="convert images ... "):
        im = Image.open(file).convert('RGB')
        temp = file.split('.')[:-1]
        temp = '.'.join(temp)
        temp = "{}.jpg".format(temp)
        im.save(temp, 'jpeg')


def make_train_test_dataset(directory_path):
    file_list = []
    file_list += file_io.get_all_file_path(directory_path, 'jpg')
    # file_list += file_io.get_all_file_path(directory_path, 'JPG')
    # file_list += file_io.get_all_file_path(directory_path, 'png')
    # file_list += file_io.get_all_file_path(directory_path, 'jpeg')
    # file_list += file_io.get_all_file_path(directory_path, 'bmp')

    file_io.make_list2txt(file_list, './dataset/kfood-list.txt')
    train_list, test_list = train_test_split(file_list, test_size=0.25)
    file_io.make_list2txt(train_list, './dataset/kfood-train.txt')
    file_io.make_list2txt(test_list, './dataset/kfood-test.txt')


def make_label(data_list, index=4):
    file_list = file_io.read_txt2list(data_list)
    label_list = []
    for file in file_list:
        label_list.append(file.split('/')[index])
    label_list = list(set(label_list))
    print(len(label_list))
    file_io.make_list2txt(label_list, './dataset/kfood-category-detail.txt')