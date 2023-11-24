import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class bnn_layer(nn.Module):

    def __init__(self, input_features, output_features, prior_var=1.):

        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        self.w = None
        self.b = None

        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, input):

        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)

        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)

        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()
 
        return F.linear(input, self.w, self.b)


class BNN(nn.Module):

    def __init__(self, input = 2 ,output =1,hidden_units = 4, noise_tol=.1, prior_var=1.):
        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        
        # Linear_BBB构造参数 input_features, output_features, prior_var=1.
        self.hidden = bnn_layer(input, hidden_units, prior_var=prior_var)
            
        self.hidden_2 = bnn_layer(hidden_units, hidden_units, prior_var=prior_var)
        self.hidden_3 = bnn_layer(hidden_units, hidden_units, prior_var=prior_var)
        self.out = bnn_layer(hidden_units, output, prior_var=prior_var)  # 输出层？也是贝叶斯
        self.noise_tol = noise_tol  # we will use the noise tolerance to calculate our likelihood

    def forward(self, x):

        x = torch.sigmoid(self.hidden(x))
        # x = torch.tanh(self.hidden_2(x))
        # x = torch.tanh(self.hidden_3(x))
        x = self.out(x)
        return x

    def log_prior(self):
        #
        # calculate the log prior over all the layers
        return self.hidden.log_prior + self.out.log_prior# + self.hidden_2.log_prior 

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post #+ self.hidden_2.log_post

    # samples 表示的是样本的数量

    def sample_elbo(self, input, target, samples):

        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)

        for i in range(samples):
            outputs[i] = self(input).reshape(-1)  # make predictions
            log_priors[i] = self.log_prior()  # get log prior
            log_posts[i] = self.log_post()  # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(
                target.reshape(-1)).sum()  # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        print(log_prior)
        print(log_post)
        print(log_like)
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss