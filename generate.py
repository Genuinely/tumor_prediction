
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import numpy as np
from tqdm import trange
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import dataset as ds
import torchvision
from torch.utils.tensorboard import SummaryWriter
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='logs/model.pth', help='path to model')
parser.add_argument('--log_dir', default='logs/plots', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=3, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=1, help='number of frames to predict')
parser.add_argument('--data_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=30, help='number of samples')
parser.add_argument('--N', type=int, default=100, help='number of samples')


opt = parser.parse_args()
os.makedirs('%s' % opt.log_dir, exist_ok=True)


opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

writer = SummaryWriter(opt.log_dir)

# ---------------- load the models  ----------------
tmp = torch.load(opt.model_path)
frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
prior = tmp['prior']
frame_predictor.eval()
prior.eval()
posterior.eval()
encoder = tmp['encoder']
decoder = tmp['decoder']
encoder.train()
decoder.train()
frame_predictor.batch_size = opt.batch_size
posterior.batch_size = opt.batch_size
prior.batch_size = opt.batch_size
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()

# ---------------- set the options ----------------
opt.channels = tmp['opt'].channels
opt.image_width = tmp['opt'].image_width

print(opt)


# --------- load a dataset ------------------------------------
import utils
train_loader, test_loader = ds.get_dataloaders(opt)

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()

# --------- eval funtions ------------------------------------
def make_gifs(x, idx, name, train):
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]

    for i in range(1, opt.n_eval):
        h = encoder(x_in)
        with torch.no_grad():
            h_target = encoder(x[i])[0]
        if i < opt.n_past:	
            h, skip = h
        else:
            h, _ = h
        with torch.no_grad():
            h = h
        _, z_t, _= posterior(h_target) # take the mean
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1)) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            with torch.no_grad():
                h_pred = frame_predictor(torch.cat([h, z_t], 1))
                x_in = decoder([h_pred, skip])
            posterior_gen.append(x_in)

    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    all_gen = []
    for s in trange(nsample):
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if i < opt.n_past:	
                h, skip = h
            else:
                h, _ = h
            with torch.no_grad():
                h = h
            if i < opt.n_past:
                with torch.no_grad():
                    h_target = encoder(x[i])[0]
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                with torch.no_grad():
                    h = frame_predictor(torch.cat([h, z_t], 1))
                    x_in = decoder([h, skip])
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_in)
        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)
        if train:
            train_ssim.append(np.mean(ssim))
            train_psnr.append(np.mean(psnr))
        else:
            test_ssim.append(np.mean(ssim))
            test_psnr.append(np.mean(psnr))
    ###### ssim ######
    for i in range(opt.batch_size):
        result = []
        gifs = [ [] for t in range(opt.n_eval) ]
        text = [ [] for t in range(opt.n_eval) ]
        mean_ssim = np.mean(ssim[i], 1)

        ordered = np.argsort(mean_ssim)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for t in range(opt.n_eval):
            # gt 
            result.append(x[t][i])
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append('Ground\ntruth')
            #posterior 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')
            # best 
            if t < opt.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            
            # predicted
            
            text[t].append('Best SSIM')
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))
        result.append(all_gen[sidx][3][i])
        grid = torchvision.utils.make_grid(result)
        writer.add_image('sample: '+str(i), grid, i, dataformats='CHW')
        fname = '%s/%s_%d.gif' % (opt.log_dir, name, idx+i) 
        utils.save_gif_with_text(fname, gifs, text)

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px
# writer = SummaryWriter(opt.log_dir)
train_psnr = []
train_ssim = []
test_psnr = []
test_ssim = []

for i in range(0, opt.N, opt.batch_size):
    # plot train
    train_x = next(training_batch_generator)
    make_gifs(train_x, i, 'train', True)
    # plot test
    test_x = next(testing_batch_generator)
    make_gifs(test_x, i, 'test', False)
    print(i)

print('train_psnr: ', max(train_psnr))
print('train_ssim: ', max(train_ssim))
print('test_psnr: ', max(test_psnr))
print('test_ssim: ', max(test_ssim))