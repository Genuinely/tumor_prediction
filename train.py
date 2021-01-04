import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import KLDivLoss as kl_criterion
from torch.nn import MSELoss as mse_criterion
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.metrics import PSNR, SSIM
from pytorch_lightning.loggers import TensorBoardLogger
import models.lstm as lstm_models
import models.dcgan as model
from argparse import ArgumentParser
# check point path
# load_from_checkpoint

class SVG_LP(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
        self.posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
        self.prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
        
        self.frame_predictor.apply(utils.init_weights)
        self.posterior.apply(utils.init_weights)
        self.prior.apply(utils.init_weights)
    
    def configure_optimizers(self):
        frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def training_step(self, batch, batch_idx):
        # TODO: set nets to training mode


         #-------train the frame predictor------
        
        # zero the gradients
        frame_predictor.zero_grad()
        posterior.zero_grad()
        prior.zero_grad()
        encoder.zero_grad()
        decoder.zero_grad()

        # initialize the hidden state.
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()

        # x is a sample from the batch
        x = batch[batchidx]
        mse, kld = 0.0
        
        # TODO: understand what this is doing
        for i in range(1, opt.n_past+opt.n_future):
            h = encoder(x[i-1])
            h_target = encoder(x[i])[0]
            
            # what does this do?
            # if opt.last_frame_skip or i < opt.n_past:	
            #     h, skip = h
            # else:
            #     h = h[0]

            z_t, mu, logvar = posterior(h_target)
            _, mu_p, logvar_p = prior(h)
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred = decoder([h_pred, skip])
            mse += mse_criterion(x_pred, x[i])
            kld += kl_criterion(mu, logvar, mu_p, logvar_p)
        
        # loss 
        loss = mse + kld*opt.beta
        self.manual_backward(loss)

        # optimizer step
        frame_predictor_optimizer.step()
        posterior_optimizer.step()
        prior_optimizer.step()
        encoder_optimizer.step()
        decoder_optimizer.step()

        # return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy()/(opt.n_future+opt.n_past)

    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomCrop(96)])
        ds_train = ImageFolder(os.path.join(conf['data_root'], 'train'), transform=transform_train)

        trainloader = DataLoader(ds_train, num_workers = 0, batch_size = math.ceil(batch_size / world_size), 
        sampler = sampler, shuffle = not is_ddp, drop_last = True, pin_memory = True)

        return trainloader

    def test_dataloader(self):
        transform_test = transforms.Compose([transforms.CenterCrop(96)])
        ds_test = ImageFolder(os.path.join(conf['data_root'], 'test'), transform=transform_test)
        testloader = DataLoader(ds_test, num_workers = 0, batch_size = math.ceil(batch_size / world_size), 
        shuffle = False, drop_last = False, pin_memory = True)

        return testloader

if __name__ == '__main__':
    parser = ArgumentParser()
    # trainer args
    
    # model args
    parser.add_argument('--batchsize', type=int, default=32)
    
    # program args
    
    
    
    opt = parser.parse_args()


    logger = TensorBoardLogger('logs', name='SVG-LP')

    train_svg_lp = SVG_LP()

    # one gpu, custom optimization
    trainer = pl.Trainer(gpus=1, automatic_optimization=False, fast_dev_run=True, logger=logger)
    trainer.fit(train_svg_lp)



