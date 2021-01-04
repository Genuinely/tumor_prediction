import torch
import pandas as pd
from random import randrange, seed
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tf
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from einops import rearrange
import torchvision.transforms as IT 
import random
from torchvideotransforms import video_transforms as VT
from torchvision import transforms, utils

def filter_sequence(seq_dict, length=4, only_tumor=True):
    d = dict(seq_dict)
    for k in list(d.keys()):
        entry = d[k]
        if len(entry) != length or (only_tumor and 1 not in entry):
            del d[k]
    return d

# helper plotting function
# def plot_video(video):
#     w, h = (10, 10)
#     fig = plt.figure()
#     cols = 4
#     rows = 1
#     for i in range(1, cols*rows+1):
#         frame = video[i-1].squeeze()
#         fig.add_subplot(rows, cols, i)
#         plt.imshow(frame, cmap='gray')
#     plt.show()

def get_array(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    return img

def normalize(img):
    img = img.astype(np.float32)
    img = 0.1 + img * 0.8 / 255. 
    img = img[:,:,np.newaxis].astype(np.float32)
    return img

def make_video(root_dir, seq, slice_index):
    arr = [get_array(root_dir+path) for path in seq]
    slice_seq = [img[slice_index, :, :] for img in arr]
    return [normalize(img) for img in slice_seq]

# make tumor video on the fly
class TumorDataset(Dataset):

    def __init__(self, 
                 sequences, 
                 root_dir='data/35_um_data_100x100x48_niis/',
                 slices=48, 
                transform=None):
        self.root_dir = root_dir
        self.videos = []
        self.seed_is_set = False
        self.transform = transform
        for seq in tqdm(sequences):
            for i in range(slices):
                self.videos.append(make_video(self.root_dir, seq, i))
                
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        self.set_seed(idx)
        vid = self.videos[idx]
    
        if self.transform:
            vid = self.transform(vid)
    
        sample = np.zeros((4, 64, 64, 1))
        for t in range(len(vid)):
            frame = vid[t]
            sample[t, :, :, 0] += frame.squeeze()
        
        sample[sample>1] = 1
        # sample[sample>1] = 1
        return sample

# set train and test to 2 samples for now
def get_dataloaders(opt):
    df = pd.read_csv('CRNN_file_reference.csv', index_col=0)

    # literal eval to get entry as list obj
    df.loc[:,'File'] = df.loc[:,'File'].apply(lambda x: literal_eval(x))
    df.loc[:,'Label'] = df.loc[:,'Label'].apply(lambda x: literal_eval(x))

    seq_dict = dict(zip(df['File'], df['Label']))
    
    d_all = filter_sequence(seq_dict=seq_dict, length=4, only_tumor=False)

    data = list(d_all.keys())
    train, test = train_test_split(data, test_size=.3, random_state=3)

    # data loaders
    tf_list = [VT.Resize((64, 64))]
    transforms = IT.Compose(tf_list)

    train_data = TumorDataset(sequences=train, transform=transforms)
    test_data = TumorDataset(sequences=test, transform=transforms)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.data_threads, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.data_threads, drop_last=True)

    return train_loader, test_loader