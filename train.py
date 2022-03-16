#cj -s '../../../socal' '../../../socal'

import torch
from experiment.experiment import Experiment

for i in range(1):
    dataset_opts  = {
                    'frame_res': 64,
                    'downsample_fac': 4,
                    'dataset_path': '../../../socal'
                    }

    net_opts = {
                'embed_dim': 512,
                'num_layers': 2,
                'nhead': 8,
                'pe': False,
                'dropout': 0
                }

    train_opts   = {
                    'task': 'SF',
                    'optim': 'Adam',
                    'weight_decay': 1e-4,
                    'epochs': 2,
                    'lr': 0.001,
                    'milestones': [],
                    'gamma': 0.1,
                    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    'seed': 0,
                    }

    results_opts = {
                    'training_results_path': './results',
                    'train_dump_file'   : 'training_results.json',
                    }

    opts = dict(dataset_opts, **net_opts)
    opts = dict(opts, **train_opts)
    opts = dict(opts, **results_opts)

    exp = Experiment(opts)
    exp.run()

