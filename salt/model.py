"""
This is a DNN with a Gaussian Mixture (GM) as likelihood.

Loss function: negative log-likelihood assuming that the uncertainties (sigmas from the 
Gaussians) depend on the input features, i.e. an heteroscedastic loss.
Outputs: parameters of the gaussians that form the GM, which are the mean, sigma and alpha 
(weight). We have a total of 3*N_mix, where N_mix is the numer of mixtures of the GM (hyperparameter).
Prediction: mode of the GM.
Uncertainty: resulting uncertainty of the Gaussian mixture. It is interpreted as a systematic uncertainty.

Author: Isabel Sainz (based on the work of Lorenz Vogel)

"""


# standard Python imports
import numpy as np
import os, sys
import argparse

# PyTorch imports
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


from sklearn.preprocessing import MaxAbsScaler



class HeteroscedasticDNN(nn.Module):

    def __init__(self, layers, dim_ipt, dim_opt, activation='relu'):
        super(HeteroscedasticDNN, self).__init__()

        self.layers     = layers  # width and depth of the network 
        self.dim_ipt    = dim_ipt # size/dimension of the input layer (first layer)
        self.dim_opt    = dim_opt # size/dimension of the output layer (last layer)
        self.activation = activation # activation function for the hidden layers

        # add first and last layer to list with all layers
        self.layers = [self.dim_ipt]+self.layers
        self.layers.append(self.dim_opt)

        # construct all layers (and their activation function)
        self.all_layers = []
        for i in range(len(self.layers)-1):
            layer = nn.Linear(self.layers[i], self.layers[i+1])
            self.all_layers.append(layer)
            if i < (len(self.layers)-2): # skipping last layer
                if self.activation.lower() == 'relu':
                    self.all_layers.append(nn.ReLU())
                elif self.activation.lower() == 'lrelu':
                    self.all_layers.append(nn.LeakyReLU())
                elif self.activation.lower() == 'gelu':
                    self.all_layers.append(nn.GELU())

        self.model = nn.Sequential(*self.all_layers)
        self.loss_function = nn.GaussianNLLLoss()

    def forward(self, x):
        return self.model(x)
    
class GaussianMixtureDNN(HeteroscedasticDNN):

    def __init__(self, device, layers, dim_ipt, num_mixtures, activation='relu'):
        super(GaussianMixtureDNN, self).__init__(
            layers       = layers,  # width and depth of the network 
            dim_ipt      = dim_ipt, # size/dimension of the input layer (first layer)
            dim_opt      = 3*num_mixtures, # size/dimension of the output layer (last layer)
            activation   = activation # activation function for the hidden layers
        )
        self.device       = device
        self.num_mixtures = num_mixtures
        self._mus         = None # mean value(s)
        self._sigma2s     = None # squared standard deviation(s)
        self._alphas      = None # mixture weights / pre-factors

    def forward(self, x):
        y = self.model(x) # feed topo-cluster features through network
        m = y[:,0*self.num_mixtures:1*self.num_mixtures]
        s = y[:,1*self.num_mixtures:2*self.num_mixtures]
        a = y[:,2*self.num_mixtures:]  # mixture weights
        a = nn.Softmax(dim=-1)(a) # normalize mixture weights

        #return torch.cat((y, alphas), axis=1)
        return m, s, a

    def compute_predictions(self, x):
        mus, sigma2s, alphas = self.forward(x)

        self._mus     = mus
        self._sigma2s = sigma2s.exp()
        self._alphas  = alphas
        
        return self._mus, self._sigma2s, self._alphas

    def log_likelihood(self, x, mus, log_sigma2s, alphas):
        """
        Logarithm of the probability density of a Gaussian mixture distribution (log-likelihood).
        This construction uses the log-sum-exp method from PyTorch which is numerically
        more stable then just computing: log(prob) = log(sum_i alpha_i*Gauss(mu_i, sigma_i))
        """

        # calculate one-dimensional negative log-Gaussians
        # neg_log_gauss = -log(Gauss(mu_i, sigma_i))
        # neg_log_gauss : shape = (num_monte, batch_size, x_sample, 3)
        neg_log_gauss  = torch.pow(mus-x, 2.0)/(2.0*torch.exp(log_sigma2s)) + 0.5*log_sigma2s
        neg_log_gauss += 0.5*np.log(2.0*np.pi) # constant

        # log_prob_components = -neg_log_gauss_i + log(alpha_i)
        #                     = log(Gauss(mu_i, sigma_i)) + log(alpha_i)
        # log_prob_components : shape = (num_monte, batch_size, x_sample, 3)
        log_likelihood_components = -neg_log_gauss + torch.log(alphas)
        
        # log_prob = log(sum_i exp(log_prob_components_i))
        #          = log(sum_i exp(log(Gauss(mu_i, sigma_i)) + log(alpha_i)))
        # log_prob : shape = (num_monte, batch_size, x_sample)
        log_likelihood = torch.logsumexp(log_likelihood_components, dim=-1)

        return log_likelihood
    

    def compute_losses(self, x, targets):
        loss_nll = self.neg_log_likelihood(x, targets)
        return loss_nll

    def neg_log_likelihood(self, x, targets):

        # compute output of HGM
        targets_reshaped = targets[:,None]
        #outputs = self.forward(x) # feed data through network
        mus, log_sigma2s, alphas = self.forward(x)
        #mus         = outputs[:,0:self.num_mixtures]
        #log_sigma2s = outputs[:,1*self.num_mixtures:2*self.num_mixtures]
        #alphas      = outputs[:,2*self.num_mixtures:]

        neg_log_likelihood = -self.log_likelihood(targets_reshaped, mus, log_sigma2s, alphas)
        return torch.mean(neg_log_likelihood) # average over batch of data

    
    def training_pass(self, x_feats, y_truth):
        self.train() # training mode
        self.optimizer.zero_grad()

        loss = self.compute_losses(x_feats, y_truth)

        loss.backward() # compute the gradients based on the loss
        self.optimizer.step() # update the network with new adjusted parameters
        return loss.item()
    
    def get_loss(self, x_feats, y_truth):
        self.eval() # evaluation mode
        with torch.no_grad():

            loss = self.compute_losses(x_feats, y_truth)
        return loss.item()
    
    def get_predictions(self, dataloader, prediction):        
        #y_truth, y_model, s_model = [], [], []
        y_truth, ys_model, sigmas_stoch = [], [], []
        mus, sigma2s, alphas = [], [], []
        outer_trn_mode = self.training
        if outer_trn_mode: self.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                x_feats = data[:,1:].float()
                y_truth.append(data[:,0].float())
                
                #prediction = self(x_feats)
                # compute model predictions (forward-pass features trough network)
                mu, sigma2, alpha = self.compute_predictions(x_feats)
                if   prediction.lower() == 'mode': 
                    y_model = self.modes()
                elif prediction.lower() == 'mean': 
                    y_model = self.means()
            
                sigma_stoch = self.sigma_stoch() 

                # detach, convert to NumPy and append everything into lists
                ys_model.append(y_model.cpu().detach().numpy())
                sigmas_stoch.append(sigma_stoch.cpu().detach().numpy())
                
                mus.append(mu.cpu().detach().numpy())
                sigma2s.append(sigma2.cpu().detach().numpy())
                alphas.append(alpha.cpu().detach().numpy())
        if outer_trn_mode: self.train()
        
        y_truth = torch.cat(y_truth).detach().cpu().numpy()
        ys_model = np.concatenate(ys_model)
        sigmas_stoch   = np.concatenate(sigmas_stoch)
        mus = np.concatenate(mus)
        sigma2s = np.concatenate(sigma2s)
        alphas = np.concatenate(alphas)

        return y_truth, ys_model, sigmas_stoch, mus, sigma2s, alphas
    
    def fit(self, output_path, dataloader_trn, dataloader_val, num_epochs, learning_rate, lr_milestones, val_every, save_every):
        
        self.output_path    = output_path
        self.dataloader_trn = dataloader_trn
        self.dataloader_val = dataloader_val
        self.num_epochs     = num_epochs
        self.learning_rate  = learning_rate
        self.lr_milestones  = lr_milestones
        self.val_every      = val_every
        self.save_every     = save_every
        
        # initialise lists for storing training statistics
        self.epoch_losses_trn = []
        self.epoch_losses_val = []

        # set up an optimizer: the optimizer updates the parameters of the network
        # here: Adam (adaptive moment estimation)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.lr_milestones[0] != 0:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_milestones, gamma=0.1)
        
        print(f'starting training loop, running for {self.num_epochs} epochs')
        for epoch in range(self.num_epochs):
            
            # training batch-loop: the inner loop goes through the training dataset batch by batch
            self.batch_losses_trn = []
            for i, data in enumerate(self.dataloader_trn):
                x_feats = data[:,1:].float() # features / observables
                y_truth = data[:,0].float()  # regression target (response)
                loss = self.training_pass(x_feats, y_truth)
                self.batch_losses_trn.append(loss)
            if self.lr_milestones[0] != 0:
                self.scheduler.step() # update learning-rate scheduler
            self.epoch_losses_trn.append(np.mean(self.batch_losses_trn))
            print(f'[{epoch:04d}] avg trn loss per batch: {self.epoch_losses_trn[-1]:16.13f}')
            
            # validation batch-loop
            if (not self.dataloader_val is None) and (self.val_every > 0) and (epoch % self.val_every == 0):
                self.batch_losses_val = []
                for i, data in enumerate(self.dataloader_val):
                    x_feats = data[:,1:].float() # features / observables
                    y_truth = data[:,0].float()  # regression target (response)
                    loss = self.get_loss(x_feats, y_truth)
                    self.batch_losses_val.append(loss)
                self.epoch_losses_val.append(np.mean(self.batch_losses_val))
                print(f'[info] avg val loss per batch: {self.epoch_losses_val[-1]:16.13f}')

            # save model parameters / weights
            if (self.save_every > 0) and (epoch % self.save_every == 0):
                filename = f'{self.output_path}/epoch-results/epoch_{epoch+1:03d}_model_state.pt'
                torch.save({'epoch': epoch+1, 'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, filename)

        # save model parameters / weights             
        torch.save({'epoch': epoch+1, 'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            f'{self.output_path}/final_model_state.pt')
        print(f'saved trained network')
        
        np.save(f'{self.output_path}/epoch_losses_trn.npy', self.epoch_losses_trn)
        np.save(f'{self.output_path}/epoch_losses_val.npy', self.epoch_losses_val)
        print(f'saved training and validation losses')

        return self.epoch_losses_trn, self.epoch_losses_val
    




    def modes(self):

        alphas  = self._alphas[None,:,:]
        sigma2s = self._sigma2s[None,:,:]
        mus     = self._mus[None,:,:]
        
        # select range to check for maxima
        x_min = 0.9*torch.min(mus, dim=-1)[0]#.cpu().detach().numpy()
        x_max = 1.1*torch.max(mus, dim=-1)[0]#.cpu().detach().numpy()
        x_min = torch.min(x_min, dim=0)[0].cpu().detach().numpy()
        x_max = torch.max(x_max, dim=0)[0].cpu().detach().numpy()
        
        # create array of ranges to check for maxima
        x_test = torch.from_numpy(np.linspace(x_min, x_max, num=1000)).to(self.device)
        x_test = torch.stack([x_test])
        
        # x_test  : shape = (x_sample, num_monte, batch_size)
        # mus     : shape = (num_monte, batch_size, num_mixtures)
        # sigma2s : shape = (num_monte, batch_size, num_mixtures)
        # alphas  : shape = (num_monte, batch_size, num_mixtures)

        # reshape parameters for likelihood computations
        x_test_reshaped  = torch.permute(x_test,  dims=(0, 2, 1))[:,:,:,None]
        mus_reshaped     = torch.permute(mus,     dims=(0, 1, 2))[:,:,None,:]
        sigma2s_reshaped = torch.permute(sigma2s, dims=(0, 1, 2))[:,:,None,:]
        alphas_reshaped  = torch.permute(alphas,  dims=(0, 1, 2))[:,:,None,:]

        # compute negative log-likelihood (NLL)
        neg_log_likelihood = -self.log_likelihood(x_test_reshaped, mus_reshaped, torch.log(sigma2s_reshaped), alphas_reshaped)
        neg_log_likelihood = torch.mean(neg_log_likelihood, dim=0) # mean over num_monte

        # choose x-point with largest likelihood value (= smallest NLL value)
        idxs  = torch.argmin(neg_log_likelihood, dim=-1, keepdim=True)
        modes = torch.gather(torch.squeeze(x_test_reshaped, dim=-1)[0,:,:], -1, idxs)

        # shape = (batch_size)
        predictions = torch.squeeze(modes, dim=-1)

        return predictions

    def means(self):
        # mean of Gaussian mixture distribution is given by
        # weighted mean of each individual distribution
        return torch.sum(self._alphas*self._mus, axis=-1)
    

    def sigma_stoch(self):
        """
        Averaged variance or aleatoric uncertainty
        (should capture noise on labels and input data).
        """

        # variance of a mixture model (https://en.wikipedia.org/wiki/Mixture_distribution):
        # variance(mixture) = sum_i alpha_i*var_i + sum_i alpha_i*mu_i^2 + sum_i alpha_i*mu_i
        #                   = sum_i alpha_i*var_i + sum_i alpha_i*mu_i^2 + mean
        sigma_stoch2  = torch.sum(self._alphas*self._sigma2s, axis=-1)
        sigma_stoch2 += torch.sum(self._alphas*torch.pow(self._mus, 2), axis=-1)
        sigma_stoch2 -= torch.pow(torch.sum(self._alphas*self._mus, axis=-1), 2)

        return torch.sqrt(sigma_stoch2)
    
class PTModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        m, s, a = self.model(x)
        return torch.cat([m, s, a], dim=1) #[batch, 9]
