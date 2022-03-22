import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
from PIL import Image


class SOCAL(Dataset):
    def __init__(self, 
                 train=False, 
                 frame_res=224,
                 downsample_fac=1,
                 dataset_path='../../../socal'):
        
        self.train = train
        self.downsample_fac = downsample_fac
        self.dataset_path = dataset_path
        self.images_dir = dataset_path + '/JPEGImages/' # folder containing individual frames
        self.frame_to_trial = pd.read_csv(dataset_path + '/frame_to_trial_mapping.csv') # frame to trial csv
        self.outcomes = pd.read_csv(dataset_path + '/socal_trial_outcomes.csv') # outcomes csv

        self.mean = [81.8088/255, 22.7293/255, 29.7582/255]
        self.std = [89.6872/255, 40.6592/255, 48.3578/255]
        
        resize = T.Resize(size=(frame_res, frame_res), interpolation=T.functional.InterpolationMode('nearest'))
        normalize = T.Normalize(mean=self.mean, std=self.std)
        
        self.transform = T.Compose([resize, T.ToTensor(), normalize])

        # Get unique trial ids that are in BOTH outcomes and frame_to_trial_mapping
        trial_ids_outcomes = set(self.outcomes['trial_id']) # unique trial ids in outcomes
        trial_ids_f2t = set(self.frame_to_trial['trial_id']) # unique trial ids in frame to trial
        trial_ids_both = list(trial_ids_outcomes & trial_ids_f2t) # unique trial ids in both, 143

        self.test_trial_ids = ['S203T1', 'S203T2', 'S318T1', 'S318T2', 'S807T1', # trial ids for test set, exist in BOTH outcomes and frame_to_trial_mapping
                               'S807T2', 'S303T1', 'S303T2', 'S403T1', 'S403T2',
                               'S206T1', 'S206T2', 'S316T1', 'S316T2', 'S306T1',
                               'S306T2', 'S616T1', 'S616T2', 'S505T1', 'S505T2']

        self.train_trial_ids = list(set(trial_ids_both) - set(self.test_trial_ids)) # trial ids for train set

    def __len__(self):
        if self.train:
          return len(self.train_trial_ids)  

        return len(self.test_trial_ids)

    def __getitem__(self, index):

        # get trial id associated to index
        if self.train:
          trial_id = self.train_trial_ids[index]
        else:
          trial_id = self.test_trial_ids[index]
        
        # get tth, sf, and ebl outcome data associated to trail
        # tth = outcomes[outcomes['trial_id'] == trial_id].values[0][6] # get tth
        sf = torch.tensor(self.outcomes[self.outcomes['trial_id'] == trial_id].values[0][7], dtype=torch.float32) # get sf
        ebl = torch.tensor(self.outcomes[self.outcomes['trial_id'] == trial_id].values[0][8], dtype=torch.float32) # get ebl

        # get all image filenames associated to trial id, put in list, ensure the frames are ordered
        frame_filenames = list(self.frame_to_trial[self.frame_to_trial['trial_id'] == trial_id]['frame'])
        
        if len(frame_filenames) > 60: # consider the first 60 frames
          frame_filenames = frame_filenames[:60]

        frames_list = []

        for i in range(1, len(frame_filenames), self.downsample_fac):
          frame_path = self.images_dir + frame_filenames[i] # get full path to image
          frame = Image.open(frame_path) # open image as PIL 
          frame = self.transform(frame) # apply transformation to image

          frames_list.append(frame) # list of frames of shape (3, 224, 224) ***
        
        frames = torch.stack(frames_list) # has shape (S, 3, 224, 224), S = seq length

        return frames, sf, ebl 