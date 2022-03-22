from re import T
import sys
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from os import makedirs
from utils.dump import DumpJSON
from utils.lr_scheduler import WarmUpLR
from utils.meters import AccuracyMeter, MSEMeter, AverageMeter

from dataset.socal import SOCAL
from net.aasp_model import AASP_Model

class Experiment:
    def __init__(self, opts):
        for key, value in opts.items():
            setattr(self, key, value)
    
        try:
            makedirs(self.training_results_path)
        except:
            pass
        
        # datasets and loaders
        socal_train = SOCAL(train=True, 
                            frame_res=self.frame_res, 
                            downsample_fac=self.downsample_fac, 
                            dataset_path=self.dataset_path)
        socal_test = SOCAL(train=False,
                           frame_res=self.frame_res, 
                           downsample_fac=self.downsample_fac, 
                           dataset_path=self.dataset_path)
        
        self.train_loader = DataLoader(socal_train, batch_size=1)
        self.test_loader = DataLoader(socal_test, batch_size=1)
        
        # model
        self.model = AASP_Model(enc_model=self.enc_model, 
                                enc_norm=self.enc_norm,
                                num_layers=self.num_layers,
                                num_heads=self.num_heads,
                                embed_dim=self.embed_dim,
                                norm_first=self.norm_first,
                                pe=self.pe,
                                dropout=self.dropout)
        self.model.to(self.device)
        
        # loss
        if self.task == 'SF':
            self.loss = nn.BCEWithLogitsLoss(reduction='sum')
        elif self.task == 'EBL': 
            self.loss = nn.MSELoss(reduction='sum')
        else:
            raise Exception('Task must be EBL or SF!')
        
        # optimizer and learning rate scheduler
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=self.initial_lr,
                                    betas=self.betas,
                                    weight_decay=self.weight_decay
                                    )
        
        self.lr_scheduler = WarmUpLR(optimizer=self.optimizer, 
                                     lr_max=self.lr_max, 
                                     t_warmup=self.t_warmup)
            

    def run(self):
        # seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # starts at the last epoch
        for epoch in range(1, self.epochs + 1):
            
            # json dump file
            results_path = self.training_results_path + '/' + self.train_dump_file
            results = DumpJSON(read_path=results_path,write_path=results_path)
            
            # train
            results = self.run_epoch("train",
                                     epoch,
                                     self.train_loader,
                                     results)  
            # test
            results = self.run_epoch("test",
                                       epoch,
                                       self.test_loader,
                                       results)
            
            # dump to json
            results.save()
        
        results.to_csv()    
            
    def run_epoch(self,
                  phase,
                  epoch,
                  loader,
                  results):
        
        # meters
        meters = {
            'loss': AverageMeter(),
            'accuracy': AccuracyMeter(),
            'rmse': MSEMeter(root=True)
        }
        
        # switch phase
        if phase == 'train':
            self.model.train()
        elif phase == 'test':
            self.model.eval()
        else:
            raise Exception('Phase must be train, test or analysis!')    
        
        for iter, (frames, sf, ebl) in enumerate(loader, 1):
            
            # input and target
            input = frames
            
            if self.task == 'SF':
                target = sf
            else: # EBL
                target = ebl
            
            input = input.to(self.device)
            target = target.to(self.device)

            # run model on input and compare predicted result to target
            pred = self.model(input)
            loss = self.loss(pred, target)
            
            # compute gradient and do optimizer step
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            # adjust learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                    
            # record statistics
            meters['loss'].add(float(loss.item()))
            
            if self.task == 'SF':
                meters['accuracy'].add(pred, target, input.data.shape[0])
            else: # EBL
                meters['rmse'].add(pred, target, input.data.shape[0])
            
            # append row to results CSV file
            if results is not None:
                if iter == len(loader):
                    
                    stats = {'phase': phase,
                             'epoch': epoch,
                             'iters': len(loader),
                             'iter_loss': meters['loss'].val,
                             'avg_loss': meters['loss'].avg,
                             'rmse': meters['rmse'].value(),
                             'accuracy': meters['accuracy'].value()
                             }

                    results.append(dict(self.__getstate__(), **stats))
            
        output =    '{}\t' +\
                    'Loss: {meter.val:.4f} ({meter.avg:.4f})\t' +\
                    'Epoch: [{}/{}][{}/{}]\t' \
                    .format(phase.capitalize(),
                            epoch,
                            self.epochs,
                            iter,
                            len(loader),
                            meter=meters['loss'])

        print(output)
        sys.stdout.flush()
                    
        return results
    
    
    def __getstate__(self):
        state = self.__dict__.copy()
        
        # remove fields that should not be saved
        attributes = [
                      'train_loader',
                      'test_loader',
                      'model',
                      'loss',
                      'optimizer',
                      'lr_scheduler',
                      'dataset_path',
                      'device',
                      'seed'
                      ]
        
        for attr in attributes:
            try:
                del state[attr]
            except:
                pass
        
        return state
    
