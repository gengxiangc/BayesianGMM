# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:33:36 2020

@author: Wangjin

BayesianGaussionMixtureModel

Stochastic variational inference

"""

import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence

import GMM


class GaussianMixtureModel():
# 3 mixture components of D=2      
    def __init__(self, Nc, Nd):
        '''
        Nc : number of components
        Nd : number of dimension
        '''
        # Initialize
        super(GaussianMixtureModel, self).__init__()
        self.Nc = Nc
        self.Nd = Nd
        
        # Variational distribution variables for means: u ~ Normal (locs, scales)
        self.locs   = Variable( torch.normal(10*torch.zeros((Nc,Nd)),1), requires_grad=True)
        self.scales = Variable( torch.pow(Gamma(5, 5).rsample((Nc, Nd)), -0.5) , requires_grad=True)# ??
        
        # VDV for standard deviations : sigma ~ Gamma(alpla, beta)
        self.alpha = Variable( torch.rand(Nc, Nd)*2 + 4 , requires_grad=True)# 4 is hyperparameters
        self.beta  = Variable( torch.rand(Nc, Nd)*2 + 4 , requires_grad=True)# 4 is hyperparameters
        
        # VDV for component weights: theta ~ Dir(C)
        self.couts = Variable( 2*torch.ones((Nc,)) , requires_grad=True)# 2 is hyperparameters 
                
        # Prior distributions for the means
        self.mu_prior = Normal(torch.zeros((Nc, Nd)), torch.ones((Nc, Nd)))
        
        # Prior distributions for the standard deviations
        self.sigma_prior = Gamma(5*torch.ones((Nc, Nd)), 5*torch.ones((Nc, Nd)))
        
        # Prior distributions for the components weights
        self.theta_prior = Dirichlet(5*torch.ones((Nc, ))) # uniform 0.2 * 5
        
    def train(self, x, sampling=True, independent=True):
        '''
        Parameters
        ----------
        x : a batch of data
        sampling : whether to sample from the variational posterior
        distributions(if Ture, the default), or just use the mean of
        the variational distributions
        
        Return
        ------
        log_likehoods : log like hood for each sample
        kl_sum : Sum of the KL divergences between the variational
            distributions and their priors
        '''
        
        # The variational distributions
        mu    = Normal(self.locs, self.scales)
        sigma = Gamma(self.alpha, self.beta)
        theta = Dirichlet(self.couts)
        
        # Sample from the variational distributions
        if sampling:
#            Nb = x.shape[0]
            Nb = 1
            mu_sample    = mu.rsample((Nb,))
            sigma_sample = torch.pow(sigma.rsample((Nb,)), -0.5)
            theta_sample = theta.rsample((Nb,))
        else:
            mu_sample = torch.reshape(mu.mean, (1, self.Nc, self.Nd))
            sigma_sample = torch.pow(torch.reshape(sigma.mean, (1, self.Nc, self.Nd)), -0.5)
            theta_sample = torch.reshape(theta.mean, (1, self.Nc)) # 1*Nc
                
        # The mixture density
        log_var = (sigma_sample **2).log()
        log_likelihoods = GMM.get_likelihoods(x, mu_sample.reshape((self.Nc,self.Nd)),
                                              log_var.reshape((self.Nc,self.Nd)), 
                                              log=True) # Nc*Nb
        
        log_prob_ = theta_sample @ log_likelihoods
        log_prob = log_prob_
        
       
        # Compute the KL divergence sum
        mu_div    = kl_divergence(mu,    self.mu_prior)   
        sigma_div = kl_divergence(sigma, self.sigma_prior) 
        theta_div = kl_divergence(theta, self.theta_prior) 
        KL = mu_div + sigma_div + theta_div
        if 0:
            print("mu_div: %f \t sigma_div: %f \t theta_div: %f"%(mu_div.sum().detach().numpy(), 
                                                              sigma_div.sum().detach().numpy(), 
                                                              theta_div.sum().detach().numpy()))
        return  KL, log_prob
                
        
if __name__ == '__main__':  
    # Generate some data
    # k = Variable(torch.tensor([1]), requires_grad=True)
    N = 3000
    X = np.random.randn(N, 2).astype('float32')
    X[:1000, :] += [2, 0]
    X[1000:2000,:] -=[2,4]
    X[2000:, :] +=[-2, 4]
    
    batch_size = 100
    loader = torch.utils.data.DataLoader(X,
                            batch_size = batch_size,
                            shuffle = True)
    # Plot the data
#    plt.plot(X[:, 0], X[:, 1], '.', alpha=0.2)
#    plt.axis('equal')
#    plt.show()
    
    model = GaussianMixtureModel(3, 2)
    opt_SGD = torch.optim.SGD([model.locs,
                          model.scales,
                          model.alpha,
                          model.beta,
                          model.couts], lr = 1e-3) #Adam
    for epoch in range(5):
        for batch in enumerate(loader):
            KL_sum , log_prob = model.train(batch[1], sampling=True)
            KL_sum = KL_sum.sum()
            log_prob = log_prob.mean()
            l =  KL_sum - log_prob
            opt_SGD.zero_grad()
            l.backward()
            opt_SGD.step()
            print("Epoch: %d \t KL: %f \t log_prob: %f"%(epoch, KL_sum.detach().numpy(), log_prob.detach().numpy()))
            print(model.locs)
                                                                  
    # Compute log likelihood at each point on a grid
    Np = 100 #number of grid points
    Xp, Yp = np.meshgrid(np.linspace(-6, 6, Np), np.linspace(-6, 6, Np))
    Pp = np.column_stack([Xp.flatten(), Yp.flatten()]).astype('float32')
    _, Z = model.train(torch.tensor(Pp), sampling=False)
    Z = Z.detach().numpy()
    Z = np.reshape(Z, (Np, Np))
            
    # Show the fit mixture density
    plt.imshow(np.exp(Z),
               extent=(-6, 6, -6, 6),
               origin='lower')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Likelihood')        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        