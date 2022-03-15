import argparse
import os
import torch
import json
import numpy as np
import src.utils.interface_train_tool as train_tool
import src.trainers.trainer as trainer
import src.trainers.tester as tester
import src.utils.interface_tensorboard as tensorboard
import src.data.dataset as dataset
import src.models.model as model_pack
import src.optimizers.optimizer as optimizer_pack
import src.optimizers.loss as loss
from src.utils import interface_file_io as file_io
import PIL
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    parser = argparse.ArgumentParser(description='waverdeep - Food Recognition')
    parser.add_argument("--configuration", required=False,
                        default='./config/config-ResNET152Combine-INGD_V2.json',)
    parser.add_argument("--image", required=False, default='./dataset/INGD_V2/무말랭이/222.jpg')
    parser.add_argument('--label', required=False, default='./config/labels.txt')
    args = parser.parse_args()

    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    print(">> load model ...")
    model = model_pack.load_model(config, checkpoint_path=config['load_checkpoint'])

    print(">> load image ...")
    input_data = PIL.Image.open(args.image)
    print(">> load label list ... ")
    label_list = file_io.read_txt2list(args.label)

    if config['use_cuda']:
        model = model.cuda()

    out = tester.inference(config, model, input_data)

    if config["use_cuda"]:
        out = out.cpu()
    print(out)
    return label_list[out]


if __name__ == '__main__':
    result = main()
    print(result)





