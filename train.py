#cj -s './socal' '/scratch/jmackey/CJRepo/socal'

import torch
from experiment.experiment import Experiment


dataset_opts  = {
                'frame_res': 224,
                'downsample_fac': 1,
                'dataset_path': './socal'
                }

net_opts = {
            'embed_dim': 512,
            'num_layers': 6,
            'nhead': 8,
            'pe': False,
            'dropout': 0
            }

train_opts   = {
                'task': 'SF',
                'optim': 'Adam',
                'weight_decay': 1e-4,
                'epochs': 20,
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
opts = dict(opts, results_opts)

exp = Experiment(opts)
exp.run()

