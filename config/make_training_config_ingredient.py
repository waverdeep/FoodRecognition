import json

configuration = {
    # definition
    "use_cuda": True,
    "epoch": 800,
    "batch_size": 16,
    "learning_rate": 0.001,
    # dataset
    "dataset_type": "INGDDataset",
    'dataset_name': 'INGD_V1',
    "train_dataset": "./dataset/INGD_V1-train.txt",
    "test_dataset": "./dataset/INGD_V1-test.txt",
    "num_workers": 8,
    "dataset_shuffle": True,
    "pin_memory": False,
    "crop_size": 512,
    # model
    "model_name": "ResNET152Combine",
    'last_node': 2048,
    'num_classes': 58,
    'model_checkpoint': None,
    # optimizer
    "optimizer_name": "Adam",
    "weight_decay": 0,
    "eps": 1e-08,
    "amsgrad": False,
    "betas": (0.9, 0.999),
    # checkpoint
    "checkpoint_save_directory_path": "./checkpoint",
    "load_checkpoint": '',
}

if __name__ == '__main__':
    name = "{}-{}".format(
        configuration['model_name'],
        configuration['dataset_name']
    )

    configuration["log_filename"] = "./log/{}".format(name)
    configuration["tensorboard_writer_name"] = "./runs/{}".format(name)
    configuration["checkpoint_file_name"] = "{}".format(name)

    filename = 'config-{}.json'.format(name)
    with open('./{}'.format(filename), 'w', encoding='utf-8') as config_file:
        json.dump(configuration, config_file, indent='\t')
