# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:34:40 2020

@author: Wangjin
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import tqdm
import sys

sys.path.append("..")


def logsumexp(x, dim, keepdim=False):
    """
    :param x:
    :param dim:
    :param keepdim:
    :return:
    """
    max, _ = torch.max(x, dim=dim, keepdim=True)
    out = max + (x - max).exp().sum(dim=dim, keepdim=keepdim).log()
    return out

def sample(mu, var, nb_samples=500):
    '''
    Parameters
    ----------
    mu: torch.Tensors,  nb_features
    var: torch.Tensors, nb_features
    
    Return
    ------
    torch.Tensor : (nb_samples, nb_features)
    '''
    out = []
    for i in range(nb_samples):
        out += [torch.normal(mu, var.sqrt())]
                
    return torch.stack(out, dim=0)

def initialize(data, K, var =1):
    '''
    Parameters
    ----------
    data: nb_samples * nb_features
    K : number of Gaussians
    param var: initial variance
    '''
    # Choose K points randomly as centers
    m = data.size(0)
    idxs = np.random.choice(m, k, replace=False)
    mu = data[idxs]
    
    # Uniform sampling for means and variances
    d = data.size(1)
    var = torch.Tensor(K, d).fill_(var)
    
    # Uniform prior: latent variables z
    pi = torch.empty(K).fill_(1. /K)
    
    return mu, var, pi

def log_gaussian(x, mean=0, logvar=0.):
    '''
    Returns the density of x under the supplied gaussian
    Defaults to standard gaussian N(0, I)
    Parameters
    ----------
    x
    '''
    if type(logvar)=='float':
        logvar = x.new(1).fill_(logvar)
    a = (x - mean) ** 2
    log_p= -0.5*(np.log(2*np.pi) + logvar + a / logvar.exp())
    
    return log_p

def get_likelihoods(X, mu, logvar, log=True):
    '''
    Parameters
    ----------
    X: nb_samples, nb_features
    logvar : log-variances: K * features
    
    Returns
    -------
    likelihoods : K, nb_samples
    '''
    
    # Get feature-weise log-likelihood : K , nb_samples, nb_features
    log_likelihoods = log_gaussian(
            X[None, :, :], # (1, nb_samples, nb_features)
            mu[:, None, :], # (K, 1, nb_features)
            logvar[:, None, :], # (K, 1, nb_features)
            )
    
    # Sum over features
    log_likelihoods = log_likelihoods.sum(-1) # Notice sum not mean
    
    if not log:
        log_likelihoods.exp_() 
    
    return log_likelihoods

def get_density(mu, logvar, pi, N=50, X_range=(0, 5), Y_range=(0, 5)):
    """ Get the mesh to compute the density on. """
    X = np.linspace(*X_range, N)
    Y = np.linspace(*Y_range, N)
    X, Y = np.meshgrid(X, Y)
    
    # get the design matrix
    points = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    points = torch.from_numpy(points).float()
    
    # compute the densities under each mixture
    P = get_likelihoods(points, mu, logvar, log=False)
    
    # sum the densities to get mixture density
    Z = torch.sum(P, dim=0).data.numpy().reshape([N, N])
    
    return X, Y, Z    
    
def get_posteriors(log_likelihoods, log_pi):
    '''
    Calculate the posterior log p(z|x), assuming a uniform prior p(z)
    
    Parameters
    ----------
    likelihoods: the relative likelihoods p(x|z): (K, nb_samples)
    
    Return
    ------
    posteriors: (K, nb_samples)
    '''
#    posteriors = log_likelihoods - log_likelihoods.sum(0) # + log_pi[:, None]
    posteriors = log_likelihoods # + log_pi[:, None]
    posteriors = posteriors - logsumexp(posteriors, dim=0, keepdim=True)
    return posteriors
    
    
def get_parameters(X, log_posteriors, eps=1e-6, min_var=1e-6):
    ''' #PRML P439
    X: nb_samples * nb_features
    log_posteriors : p(z|x) K * nb_samples
    '''
    
    posteriors = log_posteriors.exp()
    
    # Compute 'N_k' the proxy 'number of points' assigned to each distribution
    K = posteriors.size(0)
    N_k = posteriors.sum(1)  # 
    N_k = N_k.view(K, 1, 1)
    
    # Get the means by taking the weighted combination of points
    # (K, 1, examples) @ (1, examples, features) -> (K, 1, features)
    mu = posteriors[:, None] @ X[None,]
    mu = mu / (N_k + eps) # PRML P439
    
    # Get the new var
    temp = X - mu
    var = (posteriors[:, None] @ (temp **2))/(N_k + eps) # (K, 1, features)
    logvar = torch.clamp(var, min=min_var).log()
    
    # Get the new mixing probabilities
    pi = N_k / N_k.sum()
    
    return mu.squeeze(1), logvar.squeeze(1), pi.squeeze()
    
def plot_2d_sample(sample):
    sample_np = sample.numpy()
    x = sample_np[:, 0]
    y = sample_np[:, 1]
    plt.scatter(x, y)    
    
def plot_density(X, Y, Z, i=0):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
#    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
#                    cmap=cm.viridis) #  'viridis', 'plasma', 'inferno', 'magma'
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

    # adjust the limits, ticks and view angle
    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)
#     plt.savefig('fig_{}.png'.format(i), dpi=400, bbox_inches='tight')
    plt.show()
 
if __name__ == '__main__': 
    # generate some clusters    
    cluster1 = sample(
        torch.Tensor([2.5, 2.5]),
        torch.Tensor([1.2, .8]),
        nb_samples=500
    )
    
    cluster2 = sample(
        torch.Tensor([7.5, 7.5]),
        torch.Tensor([.75, .5]),
        nb_samples=500
    )
    
    cluster3 = sample(
        torch.Tensor([8, 1.5]),
        torch.Tensor([.6, .8]),
        nb_samples=1000
    )    
        
     
     # create the dummy dataset, by combining the clusters.
    X = torch.cat([cluster1, cluster2, cluster3])
    plot_2d_sample(X)   
        
    # Train  
    k = 3
    d = 2
    nb_iters = 5
    data = X
    mu, var, pi = initialize(data, k, var=1)
    logvar = var.log()
    thresh = 1e-5
    prev_cost = float('inf')
    for i in range(nb_iters):
        # get the likelihoods p(x|z) under the parameters
        log_likelihoods = get_likelihoods(X, mu, logvar, log=True)
        
        # plot
        plot_density(*get_density(mu, logvar, pi, N=100, X_range=(-2, 12), Y_range=(-2, 12)), i=i)
        
        # Get Posteriors
        log_posteriors = get_posteriors(log_likelihoods, pi.log())
        
        # Cost and Convergence
        cost = log_likelihoods.mean()
        diff = prev_cost - cost
        if torch.abs(diff).item() < thresh:
            break
        prev_cost = cost
        print('Cost : ', cost)
        # Updata Parameters
        mu, logvar, pi = get_parameters(X, log_posteriors, eps=1e-6, min_var=1e-6)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    