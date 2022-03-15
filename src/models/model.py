import src.models.model_baseline as model_baseline
import torch


def load_model(config, checkpoint_path=None):
    model_name = config['model_name']
    network = None
    if model_name == 'VGG16Combine':
        network = model_baseline.VGG16Combine
    elif model_name == 'ResNET50Combine':
        network = model_baseline.ResNET50Combine
    elif model_name == 'ResNET152Combine':
        network = model_baseline.ResNET152Combine
    elif model_name == 'WideResNET50_2Combine':
        network = model_baseline.WideResNET50_2Combine
    elif model_name == 'MobileNetV2Combine':
        network = model_baseline.MobileNetV2Combine
    elif model_name == 'DenseNet121Combine':
        network = model_baseline.DenseNet121Combine
    elif model_name == 'SqueezeNet10Combine':
        network = model_baseline.SqueezeNet10Combine
    elif model_name == 'EfficientNetB4Combine':
        network = model_baseline.EfficientNetB4Combine

    model = network(
        last_node=config['last_node'],
        num_classes=config['num_classes'],
    )

    if config['model_checkpoint'] is not None:
        print('>> load checkpoints ...')
        device = torch.device('cpu')
        checkpoint = torch.load(config['model_checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model
