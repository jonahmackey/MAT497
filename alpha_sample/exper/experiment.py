import sys
import torch
import random
import torch.nn as nn
import torch.optim as optim

from os import makedirs
from loader.constructor import get_loader
from net.network import Network
from utils.dump import DumpJSON
from utils.lr_scheduler import MultiStepLR
from utils.average_meter import AverageMeter

class Experiment:
    def __init__(self, opts):
        for key, value in opts.items():
            setattr(self, key, value)
    
        try:
            makedirs(self.training_results_path)
        except:
            pass
        
        # datasets and loaders
        self.train_loader = get_loader(self, 'train', drop_last=True)
        self.test_loader = get_loader(self, 'test', drop_last=False)
        
        # model
        self.model = Network().construct(self.net, self)
        self.model.to(self.device)
        
        # loss
        func = getattr(nn, self.crit)
        self.criterion = func()
        
        # optimizer and learning rate schedualer
        func = getattr(optim, self.optim)
        self.optimizer = func(self.model.parameters(),
                              lr=self.lr,
                              **self.optim_kwargs)
        
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=self.gamma)
            

    def run(self, stats_meter, stats_no_meter):
        # seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # starts at the last epoch
        for epoch in range(1, self.epochs + 1):

            # adjust learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # json dump file
            results_src_old = self.training_results_path + '/results_epoch='+str(epoch - 1)
            results_src = self.training_results_path + '/results_epoch='+str(epoch)
            results = DumpJSON(read_path=(results_src_old+'.json'),write_path=(results_src+'.json'))
            
            # train
            results = self.run_epoch("train",
                                     epoch,
                                     self.train_loader,
                                     stats_meter,
                                     stats_no_meter,
                                     results)  
            # test
            results = self.run_epoch("test",
                                       epoch,
                                       self.test_loader,
                                       stats_meter,
                                       stats_no_meter,
                                       results)
            
            # dump to json
            results.save()
            results.to_csv()
            
            
    def run_epoch(self,
                  phase,
                  epoch,
                  loader,
                  stats_meter,
                  stats_no_meter,
                  results):
        
        # average meters
        meters = {}
        for name, func in stats_meter.items():
            meters[name] = AverageMeter()

        # switch phase
        if phase == 'train':
            self.model.train()
        elif phase == 'test':
            self.model.eval()
        else:
            raise Exception('Phase must be train, test or analysis!')    
        
        for iter, batch in enumerate(loader, 1):
            
            # input and target
            input   = batch[0]
            target  = batch[1]
            
            if not isinstance(target,torch.LongTensor):
                target = target.view(input.shape[0],-1).type(torch.LongTensor)
            
            input = input.to(self.device)
            target = target.to(self.device)

            # run model on input and compare estimated result to target
            est = self.model(input)
            loss = self.criterion(est, target)
            
            # compute gradient and do optimizer step
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                    
            # record statistics
            for name, func in stats_meter.items():
                meters[name].update(func(locals()), input.data.shape[0])
                
            # print statistics
            output = '{}\t'                                                 \
                     'Network: {}\t'                                        \
                     'Dataset: {}\t'                                        \
                     'Epoch: [{}/{}][{}/{}]\t'                              \
                     .format(phase.capitalize(),
                             self.net,
                             self.dataset,
                             epoch,
                             self.epochs,
                             iter,
                             len(loader))
                     
            for name, meter in meters.items(): 
                output = output + '{}: {meter.val:.4f} ({meter.avg:.4f})\t' \
                                  .format(name, meter=meter)
            
            print(output)
            sys.stdout.flush()
            
            # append row to results CSV file
            if results is not None:
                if iter == len(loader):
                    
                    stats = {'phase'             : phase,
                             'dataset'           : self.dataset,
                             'epoch'             : epoch,
                             'iter'              : iter,
                             'iters'             : len(loader)}
                    
                    for name, meter in meters.items():
                        stats['iter_'+name] = meter.val
                        stats['avg_'+name]  = meter.avg
                    
                    for name, func in stats_no_meter.items():
                        stats[name] = func(locals())

                    results.append(dict(self.__getstate__(), **stats))
                    
        return results
    
    
    def __getstate__(self):
        state = self.__dict__.copy()
        
        # remove fields that should not be saved
        attributes = [
                      'train_transform',
                      'test_transform',
                      'train_loader',
                      'test_loader',
                      'model',
                      'criterion',
                      'optimizer',
                      'lr_scheduler',
                      'device',
                      ]
        
        for attr in attributes:
            try:
                del state[attr]
            except:
                pass
        
        return state
    
