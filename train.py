#cj -s '../../../socal' '../../../socal'

import torch

from experiment.experiment import Experiment


# task_list = [
#     'SF',
#     'EBL'
# ]

# Image encoder
enc_model_list = [
    'resnet18',
    # 'resnet34',
    'resnet50',
]

enc_norm_list = [
    'LN',
    'IN',
    # 'BN',
]

# enc_lr_list = [
#     0.1, 
#     0.01,
#     0
# ]

# Transformer
num_layer_list = [
    2,
    # 6, 
    8,
]

norm_first_list = [
    True, 
    False
]

# pe_list = [
#     True, 
#     False
# ]

t_warmup_list = [
    1, 
    500, 
    # 4000
]

lr_max_list = [
    1e-2,
    1e-3,
    # 1e-4
]

# dropout_list = [
#     0.0,
#     0.1,
#     0.3
# ]

for enc_model_idx in range(2): 
    for enc_norm_idx in range(2): 
        for num_layer_idx in range(2): 
            for norm_first_idx in range(2): 
                for t_warmup_idx in range(2):
                    for lr_max_idx in range(2): 

                        dataset_opts  = {
                            'frame_res': 224, 
                            'downsample_fac': 1, 
                            'dataset_path': '../../../socal' 
                        }

                        img_enc_opts = {
                            'enc_model': enc_model_list[enc_model_idx], 
                            'enc_norm': enc_norm_list[enc_norm_idx], 
                        }
                        
                        transformer_opts = {
                            'num_layers': num_layer_list[num_layer_idx], 
                            'num_heads': 8, 
                            'embed_dim': 512,
                            'norm_first': norm_first_list[norm_first_idx], 
                            'pe': False,
                            'dropout': 0.1, 
                        }

                        train_opts   = {
                            'task': 'SF', 
                            'optim': 'Adam', 
                            'betas': (0.9, 0.98), 
                            'weight_decay': 1e-4, 
                            'epochs': 300, 
                            'initial_lr': 0.0,
                            'lr_max': lr_max_list[lr_max_idx],
                            't_warmup': t_warmup_list[t_warmup_idx],
                            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                            'seed': 0,
                        }

                        results_opts = {
                            'training_results_path': './results',
                            'train_dump_file'   : 'training_results.json',
                        }

                        opts = dict(dataset_opts, **img_enc_opts)
                        opts = dict(opts, **transformer_opts)
                        opts = dict(opts, **train_opts)
                        opts = dict(opts, **results_opts)

                        exp = Experiment(opts)
                        exp.run()

