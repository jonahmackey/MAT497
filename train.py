#cj -s './socal' '/scratch/jmackey/CJRepo/socal'


import torch

from experiment.experiment import Experiment
from utils.accuracy import compute_accuracy


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

def compute_accuracy_aux(variables, k):
    return compute_accuracy(variables['pred'].data, variables['target'].data, topk=(k,))[0][0]

stats_meter    = {'top1': lambda variables: float(compute_accuracy_aux(variables, 1).item()),
                  'rmse': lambda variables: float(variables['rmse']).item(),
                  'loss': lambda variables: float(variables['loss'].item()),
                  }

stats_no_meter = {}

exp = Experiment(opts)

exp.run(stats_meter, stats_no_meter)

