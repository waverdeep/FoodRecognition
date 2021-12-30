import random
import numpy as np
import os
import torch.cuda
from datetime import datetime
import src.utils.interface_file_io as file_io


def setup_seed(random_seed=777):
    torch.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True # 연산 속도가 느려질 수 있음
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def setup_timestamp():
    now = datetime.now()
    return "{}_{}_{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)


def print_model_description(description="pretext", format_logger=None, model=None):
    format_logger.info(">>> {}_model_structure <<<".format(description))
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    format_logger.info("{} model parameters: {}".format(description, model_params))
    format_logger.info("{}".format(model))


def setup_config(configuration):
    return file_io.load_json_config(configuration)


def setup_distributed_learning(config, format_logger):
    pass


def make_target(speaker_id, speaker_dict):
    targets = torch.zeros(len(speaker_id)).long()
    for idx in range(len(speaker_id)):
        targets[idx] = speaker_dict[speaker_id[idx]]
    return targets


def save_checkpoint(config, model, optimizer, loss, epoch, format_logger, mode="best", date=""):
    if mode == "best":
        file_path = os.path.join(config['checkpoint_save_directory_path'],
                                 config['checkpoint_file_name'] + "-model-best-{}.pt".format(date))

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "loss": loss}, file_path)
    format_logger.info("saved best checkpoint ... ")
