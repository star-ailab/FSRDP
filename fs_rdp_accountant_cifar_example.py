import argparse
import itertools
import numpy as np
import pandas as pd
import os
import pickle
import random
import time
import math

from scipy.special import comb as comb

import warnings

import matplotlib.pyplot as plt


import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.optim.optimizer import required
from torch.autograd import Variable
from torch.autograd import Function

from bayesian_privacy_accountant import BayesianPrivacyAccountant

from opacus import GradSampleModule
from typing import List, Tuple, Union

from torch.nn.utils._per_sample_grad import call_for_per_sample_grads
import subprocess as sp
np.set_printoptions(threshold=100000)
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    return memory_used_values[0]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='cifar10')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--batchSize', type=int, default=120, help='input batch size')
parser.add_argument('--nClasses', type=int, default=10, help='number of labels (classes)')
parser.add_argument('--nChannels', type=int, default=3, help='number of colour channels')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--C', type=float, default=3.0, help='embedding L2-norm bound, default=1.0')                                                                                                                                        
parser.add_argument('--sigma', type=float, default=6, help='noise variance, default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='output', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=4111, help='manual seed for reproducibility')
parser.add_argument('--k', type=int, default=8, help='number of D and D_prime pairs')
parser.add_argument('--diagnosis', action='store_true', help='defauls to false, verbose info is printed if set to "store_false"')
parser.add_argument('--lrDecay', action='store_true', help='schedule lr to decay"')
parser.add_argument('--pretrain', action='store_true', help='Just pretrain on CIFAR100 and exit"')
parser.add_argument('--eval_pretrain', action='store_true', help='Just pretrain on CIFAR100 and exit"')
parser.add_argument('--n_runs', type=int, default = 1, help='number of runs to average accuracy over"')

opt, unknown = parser.parse_known_args()
print("***Batch size = ", opt.batchSize)
print("***lr = ", opt.lr)
print("***sigma = ", opt.sigma)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if torch.cuda.is_available():
    opt.cuda = True
    opt.ngpu = 1
    gpu_id = 0 # default: 1
    print("Using CUDA: gpu_id = %d" % gpu_id)
    
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

    
import torch.nn.functional as F

## Some utility functions

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_w_gradsample(model):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad and p.grad_sample is not None))

def flatten_params(parameters):
    l = [torch.flatten(p) for p in parameters]
    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s+size))
        s += size
    flat = torch.cat(l).view(-1, 1)
    return flat, {"params": flat, "indices": indices}


class SimpleConvNet100_largest_gnorm(nn.Module):
    """based on Abadi et al. model"""

    def __init__(self):
        super(SimpleConvNet100_largest_gnorm, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(opt.nChannels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, opt.ndf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
            nn.GroupNorm(1, 64), ## Batch Norm is a privacy violation Remember to delete in non-private baseline and private model

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
            nn.GroupNorm(1, 128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
            nn.GroupNorm(1, 256),
            Flatten()
            )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            )

    def forward(self, x):
        x = self.convnet(x)
        x = self.classifier(x)
        return x 

def update_param_layer(p, normalizer, batch_size, sigma, lr):
    num_dims = len(p.grad_sample.data.shape)
    if (num_dims == 3):
        normalizer = normalizer.view(-1,1,1)
    elif (num_dims == 5): # convnet layer
        normalizer = normalizer.view(-1,1,1,1,1)
    clipped = p.grad_sample.data[:int(batch_size)] / normalizer ## broadcast normalizer to match grads dimensions
    ## Accumulate, add noise, and divide by batch size
    accumulated = torch.sum(clipped, dim =0)
    noise = torch.randn_like(accumulated) * sigma
    grad_new = 1./batch_size * (accumulated + noise)
    ## Update parameters with the new grad
    return (p - lr * grad_new)

def decay_lr(lr, epoch, rate): 
    if epoch==60:
        print("***Decay activated. lr = ", lr/rate)
        return lr/rate
    if epoch==80:
        print("***Decay activated. lr = ", lr/rate)
        return lr/rate
    if epoch==100:
        print("***Decay activated. lr = ", lr/rate)
        return lr/rate
    elif epoch==125:
        print("***Decay activated. lr = ", lr/rate)
        return lr/rate
    elif epoch==150:
        print("***Decay activated. lr = ", lr/rate)
        return lr/rate
    elif epoch==175:
        print("***Decay activated. lr = ", lr/rate)
        return lr/rate
    elif epoch==200:
        print("***Decay activated. lr = ", lr/rate)
        return lr/rate
    else:
        return lr

def train_nonprivate(epochs, lr, model, train_loader, test_loader):
    """train the network non-privately and save the model.""" 
    criterion = nn.CrossEntropyLoss(reduction='none')
    optim = torch.optim.AdamW(model.parameters(), lr)
    print("*** optimizer: ", optim)
    best = 0.
    for epoch in range(epochs):
        # Training Phase 
        model.cuda(gpu_id)
        model.train()
        if (opt.lrDecay):
            lr = decay_lr(lr, epoch= epoch, rate=3)
            for param_group in optim.param_groups:
                param_group["lr"] = lr  
                
        for batch in train_loader:
            data, labels = batch 
            if opt.cuda:
                data = data.cuda(gpu_id)
                labels = labels.cuda(gpu_id)
            out = model(data)
            loss = criterion(out, labels)
            loss.mean()
            loss.mean().backward()
            optim.step()
            optim.zero_grad()
        print ("loss of epoch ", epoch, " :", loss.mean().item())
        # test
        model.eval()
        if (epoch%1 ==0):
            acc = test(test_loader, model)
            if ( acc > best): 
                best = acc 
                torch.save(model.state_dict(), "./saved_models/cnn_cifar100.pt")
    print ("*** best accuracy:", best)


def test(testloader, net):
    correct = 0.0
    total = 0.0
    
    for data in testloader:
        images, labels = data
        
        if opt.cuda:
            images = images.cuda(gpu_id)
            labels = labels.cuda(gpu_id)
            
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == (labels.long().view(-1) % opt.nClasses)).sum()

    print('Accuracy on test images: %f %%' % (100 * float(correct) / total))
    return 100 * float(correct) / total


def get_eps(*, orders: Union[List[float], float], rdp: Union[List[float], float], delta: float) -> Tuple[float, float]:
    r"""Based on:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )

    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    if idx_opt == 0 or idx_opt == len(eps) - 1:
        extreme = "smallest" if idx_opt == 0 else "largest"
        warnings.warn(
            f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound."
        )
    return eps[idx_opt], orders_vec[idx_opt]


def get_D_prime(data, label, k):
    """Return a random pair from data with replacement"""
    D_j, D_r, l_j, l_r = [], [], [], []
    for i in range(k):
        j,r = random.sample(range(int(len(trainset))), 2)
        D_j.append(data[j])
        l_j.append(label[j])
        D_r.append(data[r])
        l_r.append(label[r])
    return D_j, l_j, D_r, l_r

alphas = np.asarray( [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)))

## By Jeremiah
sigma = opt.sigma
q=120./50000
r_over_sigma_tilde=2./sigma

def M_exact(k):
    M=(-1)**(k-1)*(k-1)
    
    for ell in range(2,k+1):
        M=M+(-1)**(k-ell)*comb(k, ell, exact=True)*np.exp(ell*(ell-1)*r_over_sigma_tilde**2/2)
    return M


def B_bound(m):
    if m%2==0:
        return M_exact(m)
    else:
        return M_exact(m-1)**(1/2)*M_exact(m+1)**(1/2)


def A_bound(alpha,m):
    if m%2==0:
        A=0
        for ell in range(m+1):
            A=A+comb(m,ell,exact=True)*(-1)**(m-ell)*np.exp((alpha+ell-m-1)*(alpha+ell-m)*r_over_sigma_tilde**2/2)
        
        return A
    else:
        return A_bound(alpha,m-1)**(1/2)*A_bound(alpha,m+1)**(1/2)

def R_bound(alpha,m):
    alpha_prod=alpha
    for j in range(1,m):
        alpha_prod=alpha_prod*(alpha-j)
    if 0<alpha-m<1:
        return q**m/np.math.factorial(m)*alpha_prod*(A_bound(alpha,m)+B_bound(m))

    else:
        return q**m/np.math.factorial(m)*alpha_prod*(q/(m+1)*A_bound(alpha,m)+(1-q/(m+1))*B_bound(m))
            
def H_bound(alpha,m):
    H_terms=[1.]
    alpha_prod=alpha
    for k in range(2,m):
        alpha_prod=alpha_prod*(alpha-k+1)
        H_terms.append(q**k/np.math.factorial(k)*alpha_prod*M_exact(k))
    H_terms.append(R_bound(alpha,m))
    
    H=0.
    for j in range(len(H_terms)):
       H=H+H_terms[len(H_terms)-1-j] 
    return H


def RDP_eps_one_step(alpha,m):
    if 0<alpha-m<1:
        return 1/(alpha-1)*np.log(H_bound(alpha,m+1))
    else:
        return 1/(alpha-1)*np.log(H_bound(alpha,m))


def train(trainloader, testloader, student, n_epochs=25, lr=0.0001, accountant=None):
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(student.parameters(), lr=lr)
    
    if opt.cuda:
        student = student.cuda(gpu_id)
        criterion = criterion.cuda(gpu_id)
    
    accuracies = []
    epsilons_e5 =[]
    epsilons_e6 =[]
    epsilons_e7 =[]
    epsilons_e8 =[]
    epsilons_e9 =[]
    epsilons_e10 =[]
    epsilons_e11 =[]
    epsilons_e12 =[]
    mem_uses =[]
    
    ##
    data_loader = torch.utils.data.DataLoader(trainset, batch_size = 50000, shuffle=False, num_workers=1)
    for D,y in data_loader:
        D,y = D.cuda(gpu_id), y.cuda(gpu_id)
        break
    lr = opt.lr
    batch_size = float(opt.batchSize)
    bounds = np.zeros(len(alphas))
    for epoch in range(1, opt.n_epochs + 1):  
        running_loss = 0.0
        if (opt.lrDecay):
            lr = decay_lr(lr, epoch, 3)
                
        for i in range( int(len(trainloader.dataset)/batch_size) ):
            idx = np.random.choice(50000, size=int(batch_size), replace=False)
            inputs = D[idx].squeeze(dim=0)
            labels = y[idx].squeeze(dim=0)
            
            D_js, l_js, D_rs, l_rs = get_D_prime(D, y, opt.k)
            Ds = torch.stack(D_js + D_rs)
            ls = torch.stack(l_js + l_rs)            
            
            if opt.cuda:
                inputs = inputs.cuda(gpu_id)
                labels = labels.cuda(gpu_id)

            inputs = torch.cat((inputs, Ds))
            labels = torch.cat((labels,ls))
            
            inputv = Variable(inputs)
            labelv = Variable(labels.long().view(-1) % opt.nClasses)
            
            outputs = student(inputv)
            loss = criterion(outputs, labelv)
            
            loss.sum().backward()
            running_loss += loss.mean().item()
            ## Sanity checks
            if (opt.diagnosis):
                print("Number of all trainable parameters: ", count_parameters(student))
                print ("Number of params with populated grad_sample: ", count_parameters_w_gradsample(student))
                print ("Shape of model parameters for last linear layer bias: ", student.classifier[-1].weight.grad_sample.shape)
                for name, p in student.classifier.named_parameters():
                    print(name, p.grad_sample.shape)
            grad_samples = [p.grad_sample.detach().clone() for name, p in student.classifier.named_parameters()]
            flatten_grads = [torch.flatten(g, start_dim=1) for g in grad_samples]
            grads = torch.cat(flatten_grads, dim=1)
            
            grads_batch = grads[:int(batch_size)]
            normalizer = (torch.maximum( (torch.norm(grads_batch, p= 2, dim=1)/opt.C), torch.ones(grads_batch.shape[0]).cuda(gpu_id) )).view(-1,1)
            
            ## DP-SGD: clip, add noise, update parameters
            with torch.no_grad():
                for name, p in student.classifier.named_parameters():
                    if(opt.diagnosis):
                        print(name, p.size())
                    new = update_param_layer(p, normalizer, batch_size, opt.sigma * opt.C, lr)
                    p.copy_(new)
            
            for p in student.parameters():
                p.grad_sample = None
            student.zero_grad()
            ## No need for optimizer.step()
            
            bound_m4 = np.zeros(len(alphas))
            for j in range(len(alphas)):
                bound_m4[j] = RDP_eps_one_step(alphas[j], 10)
            
            bounds = bounds + bound_m4

        print("Epoch: %d/%d. Loss: %.3f." %
              (epoch , n_epochs, running_loss / len(trainloader)))
        
        ## Compute privacy guarantees for this epoch
        ep, alpha = get_eps(orders = alphas, rdp = bounds, delta = 1e-5)
        epsilons_e5.append(ep)
        ep, alpha = get_eps(orders = alphas, rdp = bounds, delta = 1e-6)
        epsilons_e6.append(ep)
        ep, alpha = get_eps(orders = alphas, rdp = bounds, delta = 1e-7)
        epsilons_e7.append(ep)
        ep, alpha = get_eps(orders = alphas, rdp = bounds, delta = 1e-8)
        epsilons_e8.append(ep)
        ep, alpha = get_eps(orders = alphas, rdp = bounds, delta = 1e-9)
        epsilons_e9.append(ep)
        ep, alpha = get_eps(orders = alphas, rdp = bounds, delta = 1e-10)
        epsilons_e10.append(ep)
        ep, alpha = get_eps(orders = alphas, rdp = bounds, delta = 1e-11)
        epsilons_e11.append(ep)
        ep, alpha = get_eps(orders = alphas, rdp = bounds, delta = 1e-12)
        epsilons_e12.append(ep)
        print ("** (epsilon, alpha) for this epoch: ", get_eps(orders = alphas, rdp = bounds, delta = 1e-5))
        
        ## log gpu memory for this epoch
        mem_uses.append(get_gpu_memory())
        ##
        
        student.eval()
        acc = test(testloader, student)
        accuracies.append(acc)
        student.train()
        print("Test accuracy is %d %%" % acc)
        save_step = 100
        if (epoch + 1) % save_step == 0:
            torch.save(student.state_dict(), '%s/private_net_epoch_%d.pth' % (opt.outf, epoch + 1))
    
    print('Finished Training')
    return student.cpu(), accuracies, epsilons_e5, epsilons_e6, epsilons_e7 ,epsilons_e8, epsilons_e9, epsilons_e10, epsilons_e11, epsilons_e12, acc, mem_uses

# Transformatinos based on Abadi et al.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

if (opt.pretrain or opt.eval_pretrain):
    transform_ = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.ColorJitter(),
                 transforms.RandomCrop(size=32),
                 transforms.ToTensor(),
                 transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
                 ])

else:
    transform_ = transforms.Compose(
                [                 
                 transforms.ToTensor(),
                 transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),])


if opt.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root=opt.dataroot + os.sep + opt.dataset, train=True, download=True, transform=transform_)
    testset = torchvision.datasets.CIFAR10(root=opt.dataroot + os.sep + opt.dataset, train=False, download=True, transform=transform_)
    trainset100 = torchvision.datasets.CIFAR100(root=opt.dataroot + os.sep + opt.dataset, train=True, download=True, transform=transform_)
    testset100 = torchvision.datasets.CIFAR100(root=opt.dataroot + os.sep + opt.dataset, train=False, download=True, transform=transform_)
    valset100 = torchvision.datasets.CIFAR100(root=opt.dataroot + os.sep + opt.dataset, train=False, download=True, transform=transform_)
else:
    print("Unknown dataset")
    exit(1)

# initialise data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=1, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=1)
if opt.dataset == 'cifar10':
    trainloader100 = torch.utils.data.DataLoader(trainset100, batch_size=opt.batchSize, shuffle=True, num_workers=1, drop_last=True)
    testloader100 = torch.utils.data.DataLoader(testset100, batch_size=opt.batchSize, shuffle=False, num_workers=1)
    

##
epsilon_lst =[]
runs = []
for kk in range (opt.n_runs):
    
    if (opt.pretrain):
        ## Train model on CIFAR100 non-privately
        netS100 = SimpleConvNet100_largest_gnorm()
        if (opt.diagnosis):
            summary(netS100, (3, 32, 32))
        train_nonprivate(250, opt.lr, netS100, trainloader100, testloader100)
        break
    elif (opt.eval_pretrain):
        ## Load the pretrained model
        netS = SimpleConvNet100_largest_gnorm()
        netS.load_state_dict(torch.load("./saved_models/cnn_cifar100.pt"))
        # Freeze feature extraction layers
        for name, p in netS.named_parameters():
            p.requires_grad = False
        netS.classifier[-1] = nn.Linear(512, 10)
        last_layers = [p for p in netS.parameters()][-6:]
        for ll in last_layers:
            ll.requires_grad = True
        train_nonprivate(250, opt.lr, netS, trainloader, testloader)
        break
    else:
        ## Train mdel on CIFAR10 privately
        start = time.time()
        ## Load the pretrained model
        netS = SimpleConvNet100_largest_gnorm()
        netS.load_state_dict(torch.load("./saved_models/cnn_cifar100.pt"))
        # Freeze feature extraction layers
        for name, p in netS.named_parameters():
            p.requires_grad = False
        netS.classifier[-1] = nn.Linear(512, 10)
        last_layers = [p for p in netS.parameters()][-6:]
        for ll in last_layers:
            ll.requires_grad = True
        
        if (opt.diagnosis):
            summary(netS, (3, 32, 32))
        
        netS = GradSampleModule(netS)
        total_steps = opt.n_epochs * len(trainloader)
        
        netS, accs, epsilons_e5, epsilons_e6, epsilons_e7 ,epsilons_e8, epsilons_e9, epsilons_e10, epsilons_e11, epsilons_e12, acc, mem_uses = train(trainloader, testloader, netS, lr=0.001, n_epochs=opt.n_epochs)
        
        print('e_5', repr(epsilons_e5))
        print('e_6', repr(epsilons_e6))
        print('e_7', repr(epsilons_e7))
        print('e_8', repr(epsilons_e8))
        print('e_9', repr(epsilons_e9))
        print('e_10', repr(epsilons_e10))
        print('e_11', repr(epsilons_e11))
        print('e_12', repr(epsilons_e12))
        
        print('accuracies: ', repr(accs))
        print('memory:', repr(mem_uses))
        
        plt.plot(epsilons, accs , 'black', marker='1')
        
        plt.xlabel(r'$\epsilon$', fontsize=12, rotation =0)
        plt.ylabel(r'accuracy', fontsize=12)
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        
        plt.show()
        stop = time.time()
        print("Time elapsed: %f" % (stop - start))
        
        runs.append(acc)
        
        stop = time.time()
        print("Time elapsed: %f" % (stop - start))
        
print ("*** Average performance: ", np.mean(runs))    

