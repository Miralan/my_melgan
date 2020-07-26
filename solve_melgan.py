# Load modules
from utils import getfilepath, getwavfile
from dataloader import AudioDataset, Audio2Mel_V
from model_module import Generator, Discriminator

# Load torch packages
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn

# load standard packages
import numpy as np
import time
import argparse
from pathlib import Path


def train(config):

    p = Path(config.save_path)
    p.mkdir(parents=True, exist_ok=True)
    """
    Use GPU or CPU
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """
    Initialize Generator and Dircriminator
    """
    netG = Generator(config.n_mel, config.ngf, config.n_residual_layers).to(device)
    netD = Discriminator(config.num_D, config.ndf, config.n_layers_D, config.downsamp_factor).to(device)
    # print(netG)
    # print(netD)

    """
    Initialize optimizers of netG and netD
    """
    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

    """
    Initialize dataloaders
    """
    train_set = AudioDataset(config.load_path, config.seq_len, sampling_rate=16000)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    #test_set
    #test_loader

    """
    Define cost list and time
    """
    costs = []
    start = time.time()
    fft = Audio2Mel_V().to(device)

    """
    For training faster
    """
    cudnn.enabled = True
    cudnn.benchmark = True

    """
    Training
    """
    best_mel_reconst = 10000000
    steps = 0
    for epoch in range(1, config.epochs + 1):
        for iter, (x_t, x_mel) in enumerate(train_loader):
            x_t = x_t.to(device)
            x_mel = x_mel.to(device)
            x_pred_t = netG(x_mel)

            with torch.no_grad():
                x_pred_mel = fft(x_pred_t.detach())
                x_error = F.l1_loss(x_mel, x_pred_mel).item()

            """
            Train Discriminator
            """
            D_fake_det = netD(x_pred_t.to(device).detach())
            D_real = netD(x_t.to(device))

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            netD.zero_grad()
            loss_D.backward()
            optD.step()

            """
            Train Generator
            """
            D_fake = netD(x_pred_t.to(device))

            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()

            loss_feat = 0
            feat_weights = 4.0 / (config.n_layers_D + 1)
            D_weights = 1.0 / config.num_D
            wt = D_weights * feat_weights
            for i in range(config.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

            netG.zero_grad()
            (loss_G + config.lambda_feat * loss_feat).backward()
            optG.step()

            """
            Save model, record loss information and update steps
            """
            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), x_error])

            steps = steps + 1

            if steps % config.save_interval == 0:
                st = time.time()
                """
                Generate some sample
                """
                """
                Save models and optimizers
                """
                print("############# Saveing models #############")
                torch.save({
                            'modelG': netG.state_dict(),
                            'optG': optG.state_dict(),
                            'modelD': netD.state_dict(),
                            'optD': optD.state_dict(),
                            }
                           , p.joinpath("checkpoint_{:0>6d}.pth".format(steps)))
                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(netD.state_dict(), p.joinpath("best_netD.pt"))
                    torch.save(netG.state_dict(), p.joinpath("best_netG.pt"))
                print("############ Save model Done! ################")

            if steps % config.log_interval == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iter,
                        len(train_loader),
                        1000 * (time.time() - start) / config.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()