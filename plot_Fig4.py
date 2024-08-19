#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from scipy.special import comb as comb
import matplotlib.pyplot as plt

from typing import List, Tuple, Union
import warnings

import FS_RDP_bounds as fsrdp

def K_Wang(alpha):
    K_terms=[1.]
    alpha_prod=alpha*(alpha-1)
    
    K_terms.append(2*alpha_prod*q**2*(np.exp(4/sigma**2)-1))
    
    for j in range(3,alpha+1):
        alpha_prod=alpha_prod*(alpha-j+1)
        K_terms.append(2*q**j*alpha_prod/np.math.factorial(j)*np.exp((j-1)*2*j/sigma**2))

    K=0
    for j in range(len(K_terms)):
        K=K+K_terms[len(K_terms)-1-j] 
    K=np.log(K)
    return K

def Wang_et_al_upper_bound(alpha):
    if alpha>=2:
        if int(alpha)==alpha:
            return 1./(alpha-1.)*K_Wang(alpha)
        else:
            return (1.-(alpha-math.floor(alpha)))/(alpha-1)*K_Wang(math.floor(alpha))+(alpha-math.floor(alpha))/(alpha-1)*K_Wang(math.floor(alpha)+1)
    else:
        return Wang_et_al_upper_bound(2)

def Wang_et_al_lower_bound(alpha):
    if int(alpha)==alpha:
        L_terms=[1.]
        L_terms.append(alpha*q/(1-q))
        alpha_prod=alpha
        for j in range(2,alpha+1):
            alpha_prod=alpha_prod*(alpha-j+1)
            L_terms.append(alpha_prod/np.math.factorial(j)*(q/(1-q))**j*np.exp((j-1)*2*j/sigma**2))
        
        
        L=0
        for j in range(len(L_terms)):
            L=L+L_terms[len(L_terms)-1-j]         
        return alpha/(alpha-1)*np.log(1-q)+1/(alpha-1)*np.log(L)
    else:
        print("Error, alpha must be an integer.")


def get_eps(*, orders: Union[List[float], float], rdp
            : Union[List[float], float], delta: float) -> Tuple[float, float]:
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
    multiple RDP orders and target ``delta``.
    The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
    is based on the theorem presented in the following work:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of epsilon and optimal order alpha.
    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
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
            f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound. delta is: {delta}"
        )
    return eps[idx_opt], orders_vec[idx_opt]


q=120./50000.

sigma=6.

r_over_sigma_tilde=2./sigma

N_alpha=500

alpha_array=1+10**np.linspace(-1,1.5,N_alpha)

m_array=[4]
N_m=len(m_array)

deltas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8,1e-9, 1e-10,1e-11,1e-12]
n_epochs = 250
n_train = 50000
batch_size = 120
accu_factor = n_train / batch_size * n_epochs

eps_array=np.zeros((N_m,N_alpha))

for j1 in range(N_m):
    for j2 in range(N_alpha):
        eps_array[j1,j2]=fsrdp.FSwoR_RDP_ro(alpha_array[j2], sigma, m_array[j1], q)

eps_array_poisson = np.zeros((N_m,N_alpha))

for j1 in range(N_m):
    for j2 in range(N_alpha):
        eps_array_poisson[j1,j2]=fsrdp.Poisson_RDP_ro(alpha_array[j2], sigma, m_array[j1], q)


Wang_eps_array=np.zeros(N_alpha)
for j1 in range(N_alpha):
    Wang_eps_array[j1]=Wang_et_al_upper_bound(alpha_array[j1])
    
### plot guarantees

plt.figure()
labels=[]
for k in range(N_m):
    eps_ours = []
    rdp = eps_array[k, :] * accu_factor
    for d in deltas:
        ep, alpha = get_eps(orders = alpha_array, rdp = rdp, delta = d)
        eps_ours.append(ep)
    plt.semilogx(deltas, eps_ours, marker ='1')
    labels.append('Our Thm. 3.4 upper bound '+ '(m = {})'.format(m_array[k]))

eps_wang_up = []
for d in deltas:
    rdp = Wang_eps_array * accu_factor
    ep, alpha = get_eps(orders = alpha_array, rdp = rdp, delta = d)
    eps_wang_up.append(ep)
plt.semilogx(deltas, eps_wang_up, linestyle='--')
labels.append('Wang et al. upper bound')


##Poisson replace one    
for k in range(N_m):
    eps_poisson_ro = []
    rdp = eps_array_poisson[k, :] * accu_factor
    for d in deltas:
        ep, alpha = get_eps(orders = alpha_array, rdp = rdp, delta = d)
        eps_poisson_ro.append(ep)
    plt.semilogx(deltas, eps_poisson_ro, marker ='1')
    labels.append('Poisson Subsampled RDP')


plt.xlabel(r'$\delta$', fontsize=14)
plt.ylabel(r'$\epsilon$', fontsize=14, rotation =0)
plt.legend(labels, fontsize=12, loc='upper right')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.show()    
plt.savefig("guarantee.pdf")
    
    
    
