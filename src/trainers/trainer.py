import src.optimizers.loss as losses
import torch


def train_supervised(config, writer, epoch, model, data_loader, optimizer,):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    criterion = losses.set_criterion("CrossEntropyLoss")

    for batch_idx, (input_data, target) in enumerate(data_loader):
        if config['use_cuda']:
            input_data = input_data.cuda()
            target = target.cuda()

        prediction = model(input_data)
        loss = criterion(prediction, target)

        _, predicted = torch.max(prediction.data, 1)
        accuracy = torch.sum(predicted == target)/len(target)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += len(input_data) * loss
        total_accuracy += len(input_data) * accuracy

        if writer is not None:
            writer.add_scalar('Loss/train_step', loss, (epoch - 1) * len(data_loader) + batch_idx)
            writer.add_scalar('Accuracy/train_step', accuracy * 100, (epoch - 1) * len(data_loader) + batch_idx)

    total_loss /= len(data_loader.dataset)  # average loss
    total_accuracy /= len(data_loader.dataset)  # average acc

    if writer is not None:
        writer.add_scalar('Loss/train', total_loss, (epoch - 1))
        writer.add_scalar('Accuracy/train', total_accuracy * 100, (epoch - 1))
    return total_accuracy, total_loss


