#cj -s '../../../socal' '../../../socal'

import torch

from experiment.experiment import Experiment


# task_list = [
#     'SF',
#     'EBL'
# ]

# Image encoder
# enc_model_list = [
#     'resnet18',
#     # 'resnet34',
#     'resnet50',
# ]

# enc_norm_list = [
#     'LN',
#     'IN',
#     'BN',
# ]

# enc_lr_list = [
#     0.1, 
#     0.01,
#     0
# ]

# freeze_list = [
#     True, 
#     False
# ]

# Transformer
# num_layer_list = [
#     8,
#     # 6, 
#     16,
# ]

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
    4000
]

lr_max_list = [
    1e-2,
    # 1e-3,
    1e-3,
    5e-4
]

dropout_list = [
    0.0,
    0.1, #
]

for dropout_idx in range(2):
    for norm_first_idx in range(2): 
        for t_warmup_idx in range(3):
            for lr_max_idx in range(3): 

                dataset_opts  = {
                    'frame_res': 64, 
                    'downsample_fac': 4, 
                    'dataset_path': '../../../socal' 
                }

                img_enc_opts = {
                    'enc_model': 'resnet50',
                    'enc_norm': 'BN', 
                    'pretrained': True,
                    'freeze_base': True
                }
                
                transformer_opts = {
                    'num_layers': 8, 
                    'num_heads': 8, 
                    'embed_dim': 512,
                    'norm_first': norm_first_list[norm_first_idx], 
                    'pe': True,
                    'dropout': dropout_list[dropout_idx], 
                }

                train_opts   = {
                    'task': 'SF', 
                    'optim': 'Adam', 
                    'betas': (0.9, 0.98), 
                    'weight_decay': 1e-4, 
                    'epochs': 200, 
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

