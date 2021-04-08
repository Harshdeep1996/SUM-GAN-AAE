# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
from tqdm import tqdm, trange

from layers import Summarizer, Discriminator
from utils import TensorboardWriter


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates SUM-GAN-AAE model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):

        # Build Modules
        self.linear_compress = nn.Linear(
            self.config.input_size,
            self.config.hidden_size).cuda()
        self.summarizer = Summarizer(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.discriminator = Discriminator(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers).cuda()
        self.model = nn.ModuleList([
            self.linear_compress, self.summarizer, self.discriminator])

        if self.config.mode == 'train':
            # Build Optimizers
            self.s_e_optimizer = optim.Adam(
                list(self.summarizer.s_lstm.parameters())
                + list(self.summarizer.auto_enc.e_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr)
            self.d_optimizer = optim.Adam(
                list(self.summarizer.auto_enc.d_lstm.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.lr)
            self.c_optimizer = optim.Adam(
                list(self.discriminator.parameters())
                + list(self.linear_compress.parameters()),
                lr=self.config.discriminator_lr)

            self.model.train()

            self.writer = TensorboardWriter(str(self.config.log_dir))

    def reconstruction_loss(self, hidden_gen, hidden_orig):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""
        return torch.norm(hidden_orig - hidden_gen, p=2)

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""
        return torch.abs(torch.mean(scores) - self.config.regularization_factor)

    def prior_loss(self, mu, log_var):
        ## for the VAE, KL divergence
        ## taking the exp of the log_var to get the variance back
        return 0.5 * torch.sum(-1 + log_var.exp() + mu.pow(2) - log_var)

    def gan_loss(self, original_prob, generated_prob, uniform_prob):
        ## taking formula directly from the paper
        return torch.log(original_prob) + torch.log(1 - generated_prob) + torch.log(1 - uniform_prob)

    def train(self):
        step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            s_e_loss_history = []
            d_loss_history = []
            c_loss_history = []
            for batch_i, image_features in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):

                self.model.train()

                image_features = image_features.view(-1, self.config.input_size)
                image_features_ = Variable(image_features).cuda()

                #---- Train sLSTM, eLSTM ----#
                if self.config.verbose:
                    tqdm.write('\nTraining sLSTM and eLSTM...')

                # [seq_len, 1, hidden_size]
                original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                scores, h_mu, h_log_var, generated_features = self.summarizer(original_features)
                _, _, _, uniform_features = self.summarizer(original_features, uniform=True)

                ## Using the discriminator
                h_origin, original_prob = self.discriminator(original_features)
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)
                tqdm.write(
                    f'orig_prob: {original_prob.item():.3f}, summ_prob: {fake_prob.item():.3f}, unif_prob: {uniform_prob.item():.3f}')

                reconstruction_loss = self.reconstruction_loss(h_fake, h_origin)
                sparsity_loss = self.sparsity_loss(scores)
                prior_loss = self.prior_loss(h_mu, h_log_var)

                tqdm.write(
                    f'recon loss {reconstruction_loss.item():.3f}, sparsity loss: {sparsity_loss.item():.3f}, prior loss: {prior_loss.item():.3f}')
                s_e_loss = reconstruction_loss + sparsity_loss + prior_loss

                self.s_e_optimizer.zero_grad()
                s_e_loss.backward()
                # Gradient cliping
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                self.s_e_optimizer.step()

                s_e_loss_history.append(s_e_loss.data)

                #---- Train dLSTM (generator) ----#
                if self.config.verbose:
                    tqdm.write('Training dLSTM...')

                # [seq_len, 1, hidden_size]
                # [seq_len, 1, hidden_size]
                original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                scores, h_mu, h_log_var, generated_features = self.summarizer(original_features)
                _, _, _, uniform_features = self.summarizer(original_features, uniform=True)

                ## Using the discriminator
                h_origin, original_prob = self.discriminator(original_features)
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)
                tqdm.write(
                    f'orig_prob: {original_prob.item():.3f}, summ_prob: {fake_prob.item():.3f}, unif_prob: {uniform_prob.item():.3f}')

                reconstruction_loss = self.reconstruction_loss(h_fake, h_origin)
                g_loss = self.gan_loss(original_prob, fake_prob, uniform_prob)
                tqdm.write(f'recon loss {reconstruction_loss.item():.3f}, g loss: {g_loss.item():.3f}')

                d_loss = reconstruction_loss + g_loss

                self.d_optimizer.zero_grad()
                d_loss.backward()
                # Gradient cliping
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                self.d_optimizer.step()

                d_loss_history.append(d_loss.data)

                #---- Train cLSTM ----#
                if self.config.verbose:
                    tqdm.write('Training cLSTM...')

                original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                scores, h_mu, h_log_var, generated_features = self.summarizer(original_features)
                _, _, _, uniform_features = self.summarizer(original_features, uniform=True)

                ## Using the discriminator
                h_origin, original_prob = self.discriminator(original_features)
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)
                tqdm.write(
                    f'orig_prob: {original_prob.item():.3f}, summ_prob: {fake_prob.item():.3f}, unif_prob: {uniform_prob.item():.3f}')

                c_loss = -1 * self.gan_loss(original_prob, fake_prob, uniform_prob)

                self.c_optimizer.zero_grad()
                c_loss.backward()
                # Gradient cliping
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                self.c_optimizer.step()

                c_loss_history.append(c_loss.data)

                if self.config.verbose:
                    tqdm.write('Plotting...')

                self.writer.update_loss(reconstruction_loss.data, step, 'recon_loss')
                self.writer.update_loss(sparsity_loss.data, step, 'sparsity_loss')
                self.writer.update_loss(g_loss.data, step, 'gen_loss')
                self.writer.update_loss(prior_loss.data, step, 'prior_loss')
                self.writer.update_loss(c_loss.data, step, 'c_loss')

                self.writer.update_loss(original_prob.data, step, 'original_prob')
                self.writer.update_loss(uniform_prob.data, step, 'uniform_prob')
                self.writer.update_loss(fake_prob.data, step, 'fake_prob')

                step += 1

            s_e_loss = torch.stack(s_e_loss_history).mean()
            d_loss = torch.stack(d_loss_history).mean()
            c_loss = torch.stack(c_original_loss_history).mean()

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')
            self.writer.update_loss(s_e_loss, epoch_i, 's_e_loss_epoch')
            self.writer.update_loss(d_loss, epoch_i, 'd_loss_epoch')
            self.writer.update_loss(c_original_loss, step, 'c_original_loss')
            self.writer.update_loss(c_summary_loss, step, 'c_summary_loss')

            # Save parameters at checkpoint
            ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pkl'
            tqdm.write(f'Save parameters at {ckpt_path}')
            torch.save(self.model.state_dict(), ckpt_path)

            self.evaluate(epoch_i)


    def evaluate(self, epoch_i):

        self.model.eval()

        out_dict = {}

        for video_tensor, video_name in tqdm(
                self.test_loader, desc='Evaluate', ncols=80, leave=False):

            # [seq_len, batch=1, 1024]
            video_tensor = video_tensor.view(-1, self.config.input_size)
            video_feature = Variable(video_tensor).cuda()

            # [seq_len, 1, hidden_size]
            video_feature = self.linear_compress(video_feature.detach()).unsqueeze(1)

            # [seq_len]
            with torch.no_grad():
                scores = self.summarizer.s_lstm(video_feature).squeeze(1)
                scores = scores.cpu().numpy().tolist() 

                out_dict[video_name] = scores

            score_save_path = self.config.score_dir.joinpath(
                f'{self.config.video_type}_{epoch_i}.json')
            with open(score_save_path, 'w') as f:
                tqdm.write(f'Saving score at {str(score_save_path)}.')
                json.dump(out_dict, f)
            score_save_path.chmod(0o777)

if __name__ == '__main__':
    pass
