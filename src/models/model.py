import src.models.model_baseline as model_baseline

def load_model(config):
    model_name = config['model_name']
    model = None
    if model_name == 'VGG16Combine':
        pass

    if config['model_checkpoint'] is not None:
        pass

    return model