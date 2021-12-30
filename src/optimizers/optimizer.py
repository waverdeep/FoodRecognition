import torch.optim as optimizer
import torch.optim.lr_scheduler as scheduler


def get_optimizer(model_parameter, config):
    optimizer_name =  config['optimizer_name']
    if optimizer_name == 'Adam':
        # We use the Adam optimizer [32] with a learning rate of 2e-4
        return optimizer.Adam(params=model_parameter,
                              lr=config['learning_rate'],
                              weight_decay=config['weight_decay'],
                              eps=config['eps'],
                              amsgrad=config['amsgrad'],
                              betas=config['betas'])

    elif optimizer_name == 'SGD':
        return optimizer.SGD(params=model_parameter,
                             lr=config['learning_rate'],
                             momentum=config['momentum'],
                             dampening=config['dampening'],
                             weight_decay=config['weight_decay'],
                             nesterov=config['nesterov'])


def get_scheduler(name, wrapped_optimizer, optimizer_param):
    if name == 'LambdaLR':
        return scheduler.LambdaLR(optimizer=wrapped_optimizer, lr_lambda=optimizer_param['lr_lambda'],
                                  last_epoch=optimizer_param['last_epoch'])




