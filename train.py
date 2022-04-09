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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    parser = argparse.ArgumentParser(description='waverdeep - Food Recognition')
    parser.add_argument("--configuration", required=False,
                        default='./config/config-MobileNetV3LargeCombine-INGD_V2.json')
    args = parser.parse_args()
    now = train_tool.setup_timestamp()

    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    print(">>> Train Supervised - Food Recognition <<<")

    print(">> Use GPU: ", torch.cuda.is_available())
    print(">> Config")
    print(config)

    print(">> load dataset ...")
    # setup train/test dataloader
    train_loader, train_dataset = dataset.get_dataloader(config=config, mode='train')
    test_loader, test_dataset = dataset.get_dataloader(config=config, mode='test')

    # load model
    print(">> load model ...")
    model = model_pack.load_model(config)
    print(">> load optimizer ...")
    optimizer = optimizer_pack.get_optimizer(model_parameter=model.parameters(), config=config)

    # setup tensorboard
    print(">> set tensorboard ...")
    writer = tensorboard.set_tensorboard_writer(
        "{}-{}".format(config['tensorboard_writer_name'], now)
    )

    # model inspect code
    tensorboard.inspect_model(writer=writer, model=model, data=torch.randn(8, 3, 512, 512))
    # watch dataset on tensorboard
    tensorboard.add_image_on_tensorboard(writer, train_loader, desc='train')
    tensorboard.add_image_on_tensorboard(writer, test_loader, desc='test')

    # if gpu available: load gpu
    if config['use_cuda']:
        model = model.cuda()

    print(">> start train/test ...")
    best_loss = None
    epoch = config['epoch']
    for count in range(epoch):
        count = count + 1
        print(">> start train ... [ {}/{} epoch - {} iter ]".format(count, epoch, len(train_loader)))
        train_accuracy, train_loss = trainer.train_supervised(
            config=config, writer=writer, epoch=count, model=model, data_loader=train_loader, optimizer=optimizer)

        print(">> start test  ... [ {}/{} epoch - {} iter ]".format(count, epoch, len(test_loader)))
        test_accuracy, test_loss = tester.test_supervised(
            config=config, writer=writer, epoch=count, model=model, data_loader=test_loader)

        if best_loss is None:
            best_loss = test_loss
        elif test_loss < best_loss:
            best_loss = test_loss
            best_epoch = count
            train_tool.save_checkpoint(config=config, model=model, optimizer=optimizer,
                                       loss=test_loss, epoch=best_epoch, mode="best",
                                       date='{}'.format(now))
            print("save checkpoint at {} epoch...".format(count))

    tensorboard.close_tensorboard_writer(writer)


if __name__ == '__main__':
    main()
