import src.utils.interface_train_tool as train_tool
import src.utils.interface_logger as logger
import src.utils.interface_tensorboard as tensorboard
import src.optimizers.optimizer as optimizers
import src.losses.loss as losses
import src.data.dataset as dataset
import src.models.model as models
import argparse
import json
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    # Configuration 불러오기
    parser = argparse.ArgumentParser(description='waverdeep - food recognition learning')
    # DISTRIBUTED 사용하기 위해서는 local rank를 argument로 받아야함. 그러면 torch.distributed.launch에서 알아서 해줌
    parser.add_argument('--configuration', required=False,
                        default='./config/config_ResNet152-Combine-training01-batch8.json') ### yaml로 변경할 예정
    args = parser.parse_args()

    train_tool.setup_seed(random_seed=777)
    now = train_tool.setup_timestamp()

    # read configuration file
    with open(args.configuration, 'r') as configuration:
        config = json.load(configuration)

    # create logger
    format_logger = logger.setup_log(save_filename="{}-{}.log".format(config['log_filename'], now))

    # gpu check
    format_logger.info("GPU: {}".format(torch.cuda.is_available()))

    # print configuration 출력
    format_logger.info('configurations: {}'.format(config))
    format_logger.info('load train/test dataset ...')

    # setup train/test dataloader
    train_loader, train_dataset = dataset.get_dataloader(config=config, mode='train')
    test_loader, test_dataset = dataset.get_dataloader(config=config, mode='test')

    # load model
    format_logger.info("load_model ...")
    model = models.load_model(config)

    # setup optimizer
    optimizer = optimizers.get_optimizer(model_parameter=model.parameters(), config=config)

    # setup tensorboard
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

    # print model information
    format_logger.info(">>> model_structure <<<")
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    format_logger.info("model parameters: {}".format(model_params))
    format_logger.info("{}".format(model))

    # start training ....
    best_accuracy = 0.0
    best_loss = None
    num_of_epoch = config['epoch']
    for epoch in range(num_of_epoch):
        epoch = epoch + 1
        format_logger.info("start train ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        train_accuracy, train_loss = train(config, writer, epoch, model, train_loader, optimizer, format_logger)
        format_logger.info("start test ... [ {}/{} epoch ]".format(epoch, num_of_epoch))
        test_accuracy, test_loss = test(config, writer, epoch, model, test_loader, format_logger)

        if best_loss is None:
            best_loss = test_loss

        if test_accuracy > best_accuracy and test_loss <= best_loss:
            best_accuracy = test_accuracy
            best_epoch = epoch
            train_tool.save_checkpoint(config=config, model=model, optimizer=optimizer,
                                       loss=test_loss, epoch=best_epoch, format_logger=format_logger, mode="best",
                                       date='{}'.format(now))

    tensorboard.close_tensorboard_writer(writer)


def train(config, writer, epoch, model, train_loader, optimizer, format_logger):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    criterion = losses.set_criterion("CrossEntropyLoss")
    for batch_idx, (input_data, targets, detail_targets) in enumerate(train_loader):
        if config['use_cuda']:
            data = input_data.cuda()
            detail_targets = detail_targets.cuda()
        prediction = model(data)
        loss = criterion(prediction, detail_targets)

        _, predicted = torch.max(prediction.data, 1)
        accuracy = torch.sum(predicted == detail_targets)/len(detail_targets)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train_step', loss, (epoch - 1) * len(train_loader) + batch_idx)
        writer.add_scalar('Accuracy/train_step', accuracy * 100, (epoch - 1) * len(train_loader) + batch_idx)
        total_loss += len(data) * loss
        total_accuracy += len(data) * accuracy

    total_loss /= len(train_loader.dataset)  # average loss
    total_accuracy /= len(train_loader.dataset)  # average acc

    writer.add_scalar('Loss/train', total_loss, (epoch - 1))
    writer.add_scalar('Accuracy/train', total_accuracy * 100, (epoch - 1))
    return total_accuracy, total_loss

    # conv = 0
    # for idx, layer in enumerate(model.modules()):
    #     if isinstance(layer, torch.nn.Conv2d):
    #         writer.add_histogram("Conv/weights-{}".format(conv), layer.weight,
    #                              global_step=(epoch - 1) * len(train_loader) + batch_idx)
    #         writer.add_histogram("Conv/bias-{}".format(conv), layer.bias,
    #                              global_step=(epoch - 1) * len(train_loader) + batch_idx)
    #         conv += 1


def test(config, writer, epoch, model, test_loader, format_logger):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    criterion = losses.set_criterion("CrossEntropyLoss")
    with torch.no_grad():
        for batch_idx, (input_data, targets, detail_targets) in enumerate(test_loader):
            if config['use_cuda']:
                data = input_data.cuda()
                detail_targets = detail_targets.cuda()
            prediction = model(data)
            loss = criterion(prediction, detail_targets)

            _, predicted = torch.max(prediction.data, 1)
            accuracy = torch.sum(predicted == detail_targets) / len(detail_targets)

            total_loss += len(data) * loss
            total_accuracy += len(data) * accuracy

        total_loss /= len(test_loader.dataset)  # average loss
        total_accuracy /= len(test_loader.dataset)  # average acc

        writer.add_scalar('Loss/test', total_loss, (epoch - 1))
        writer.add_scalar('Accuracy/test', total_accuracy * 100, (epoch - 1))

        format_logger.info("[ {}/{} epoch validation result: [ average acc: {}/ average loss: {} ]".format(
            epoch, config['epoch'], total_accuracy, total_loss
        ))

    return total_accuracy, total_loss


if __name__ == '__main__':
    main()