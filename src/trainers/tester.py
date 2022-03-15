import src.optimizers.loss as losses
import torchvision.transforms as transforms
import torch


def test_supervised(config, writer, epoch, model, data_loader):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    criterion = losses.set_criterion("CrossEntropyLoss")
    with torch.no_grad():
        for batch_idx, (input_data, target) in enumerate(data_loader):
            if config['use_cuda']:
                input_data = input_data.cuda()
                target = target.cuda()

            prediction = model(input_data)
            loss = criterion(prediction, target)

            _, predicted = torch.max(prediction.data, 1)
            accuracy = torch.sum(predicted == target) / len(target)

            total_loss += len(input_data) * loss
            total_accuracy += len(input_data) * accuracy

        total_loss /= len(data_loader.dataset)  # average loss
        total_accuracy /= len(data_loader.dataset)  # average acc

        if writer is not None:
            writer.add_scalar('Loss/test', total_loss, (epoch - 1))
            writer.add_scalar('Accuracy/test', total_accuracy * 100, (epoch - 1))
    return total_accuracy, total_loss


def inference(config, model, input_data):
    # input data : (channel, width, height)
    image_transforms = transforms.Compose(
        [
            transforms.Resize(640),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    input_data = image_transforms(input_data)
    input_data = input_data.unsqueeze(0)

    if config['use_cuda']:
        input_data = input_data.cuda()

    prediction = model(input_data)
    _, predicted = torch.max(prediction.data, 1)
    return predicted[0]




