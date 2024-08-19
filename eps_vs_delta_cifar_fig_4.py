#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Comparison between our method and that of Wang et al
import math
import numpy as np
from scipy.special import comb as comb
import matplotlib.pyplot as plt

from typing import List, Tuple, Union
import warnings

q=120./50000.

sigma=6.

r_over_sigma_tilde=2./sigma

def M_exact(k):
    M=(-1)**(k-1)*(k-1)
    
    for ell in range(2,k+1):
        M=M+(-1)**(k-ell)*comb(k, ell, exact=True)*np.exp(ell*(ell-1)*r_over_sigma_tilde**2/2)
    return M


def log_comb(k,n):
    val=0
    for j in range(n):
        val=val+np.log(k-j)
    for j in range(1,n+1):
        val=val-np.log(j)
    return val

def log_factorial(k):
    val=0
    for j in range(1,k):
        val=val+np.log(j+1)
    return val



#Definition of auxiliary functions needed for all of the bounds 
def log_M_exact(sigma,k):
    if k==0:
        return 0
    elif k>=2:
        exponents=[]
        
        exponents.append(np.log(k-1))
        for ell in range(2,k+1):
            exponents.append(log_comb(k,ell)+2.*ell*(ell-1)/sigma**2)
            
        exp_max=np.max(exponents)
        
        
        M=(-1)**(k-1)*np.exp(exponents[0]-exp_max)
        j=1
        for ell in range(2,k+1):
            M=M+(-1)**(k-ell)*np.exp(exponents[j]-exp_max)
            j=j+1
        return np.log(M)+exp_max
    
def M_exact(sigma,k):
    return np.exp(log_M_exact(sigma,k))
    

def log_B_bound(sigma,m):
    if m%2==0:
        return log_M_exact(sigma,m)
    else:
        return 0.5*(log_M_exact(sigma,m-1)+log_M_exact(sigma,m+1))

def B_bound(sigma,m):
    if m%2==0:
        return M_exact(sigma,m)
    else:
        return M_exact(sigma,m-1)**(1/2)*M_exact(sigma,m+1)**(1/2)
    

    
def log_B_bound_precomputed(m,log_M_exact_2j_array):
    if m%2==0:
        return log_M_exact_2j_array[int(m/2)]
    else:
        return 0.5*(log_M_exact_2j_array[int((m-1)/2)]+log_M_exact_2j_array[int((m+1)/2)])

def B_bound_precomputed(m,log_M_exact_2j_array):
    return np.exp(log_B_bound_precomputed(m,log_M_exact_2j_array))


##Functions for computing FS-woR  bounds
#assumes that alpha\neq j for j={1,...,m-1}: this condition should be checked whenever this function is used
def log_R_bound(alpha,sigma,m,q):
    log_abs_alpha_prod=np.log(alpha)
    for j in range(1,m):
        log_abs_alpha_prod=log_abs_alpha_prod+np.log(np.abs(alpha-j))
        
    if alpha-m<=0:
        return np.log(q)*m-log_factorial(m)+np.log(1-q)*(alpha-m)+log_abs_alpha_prod+log_B_bound(sigma,m)
    
    
    else:
        alpha_ceil=np.ceil(alpha)
        
        exponents=[]
        
        exponents.append(log_B_bound(sigma,m)-log_factorial(m))
        for ell in range(int(alpha_ceil)-m+1):
            
            log_fact_ratio=0
            for j in range(ell):
                log_fact_ratio=log_fact_ratio+np.log(alpha_ceil-m-j)
                
            
            exponents.append(np.log(q)*ell+log_fact_ratio-log_factorial(m+ell)+log_B_bound(sigma,ell+m))
        exp_max=np.max(exponents)
        return np.log(q)*m+log_abs_alpha_prod+np.log(np.sum(np.exp(exponents-exp_max)))+exp_max
    
def R_bound(alpha,sigma,m,q):
    if alpha==int(alpha) and alpha<m:
        return 0
    
    else:
        return np.exp(log_R_bound(alpha,sigma,m,q))

    

def H_bound(alpha,sigma,m,q):
    H_terms=[1.]
    log_alpha_prod=np.log(alpha)
    s=1
    for k in range(2,m):
        log_alpha_prod=log_alpha_prod+np.log(np.abs(alpha-k+1))
        s=s*np.sign(alpha-k+1)
        H_terms.append(s*np.exp(np.log(q)*k-log_factorial(k)+log_alpha_prod+log_M_exact(sigma,k)))
    H_terms.append(R_bound(alpha,sigma,m,q))
    

    return np.sum(H_terms)


def log_H_bound_minus_1(alpha,sigma,m,q):
    exponents=[]
    signs=[]
    s=1
    
    log_alpha_prod=np.log(alpha)
    for k in range(2,m):
        if alpha-k+1==0: #no further terms contribute
            break  
        else:
            log_alpha_prod=log_alpha_prod+np.log(np.abs(alpha-k+1))
            s=s*np.sign(alpha-k+1)
            signs.append(s)
            exponents.append(np.log(q)*k-log_factorial(k)+log_alpha_prod+log_M_exact(sigma,k))
        
    if not(alpha==int(alpha) and alpha<m): #remainder is zero if alpha is an integer in {1,...,m-1}
        signs.append(1)
        exponents.append(log_R_bound(alpha,sigma,m,q))
    
    exp_max=np.max(exponents)

    return np.log(np.sum(signs*np.exp(exponents-exp_max)))+exp_max




#One step RDP bound for fixed-size subsampling without replacement under add/remove adjacency (Thm 3.3 with T=1)
def FS_RDP_woR_ar(alpha,sigma,m,q):
    val=log_H_bound_minus_1(alpha,sigma,m,q)
    if val<=0:
        return 1/(alpha-1.)*np.log(1+np.exp(val)) 
    else:
        return 1/(alpha-1.)*(np.log(np.exp(-val)+1)+val)
        
    
#Thm 3.1 bounds for integer alpha (Taylor remainder is zero)
def K_integer_alpha_ar(alpha,sigma,q):
    K_terms=[]
    for k in range(alpha+1):
        K_term=comb(alpha, k, exact=True)*(1-q)**(alpha-k)*q**k*np.exp(k*(k-1)*(2./sigma)**2/2)
        K_terms.append(K_term)
    
    
    
    #Sum terms in reverse order for numerical stability
    K=0
    for j in range(len(K_terms)):
        K=K+K_terms[len(K_terms)-1-j] 
    K=np.log(K)
    return K

# bounds obtained by combining formula for integer alpha with convexity technique of Wang et al
def FS_RDP_woR_ar_convexity_method(alpha,sigma,q):
    if alpha>=2:
        if int(alpha)==alpha:
            return 1./(alpha-1.)*K_integer_alpha_ar(int(alpha),sigma,q)
        else:
            return (1.-(alpha-math.floor(alpha)))/(alpha-1)*K_integer_alpha_ar(int(math.floor(alpha)),sigma,q)+(alpha-math.floor(alpha))/(alpha-1)*K_integer_alpha_ar(int(math.floor(alpha)+1),sigma,q)
    else:
        return FS_RDP_woR_ar_convexity_method(2,sigma,q)
    
    
## FS_woR RDP bounds under replace-one adjacency (Thm 3.4 bounds)

#upper bound on log of kth-derivative for k>=3
def log_F_deriv_bound(alpha,sigma,k,log_M_exact_2j_array):
    exponents=[]
    if k%2==0:
        exponents.append(np.log(4)+log_B_bound_precomputed(k,log_M_exact_2j_array))
    else:
        exponents.append(np.log(3)+log_B_bound_precomputed(k,log_M_exact_2j_array))
        
    
    
    for j in range(k+1):
        prod1=1.
        for ell in range(1,j):
            prod1=prod1*(1-ell/alpha)
        prod2=1.
        for ell in range(k-j):
            prod2=prod2*(1+(ell-1)/alpha)
        
        C=np.abs((alpha/(alpha-1))*prod1*prod2-1.)
        
        if C>0:  
            exponents.append(log_comb(k,j)+np.log(C)+log_B_bound_precomputed(k,log_M_exact_2j_array))
        
        exp_max=np.max(exponents)
            
        val=np.log(np.sum(np.exp(exponents-exp_max)))+exp_max
    return np.log(alpha-1.)+np.log(alpha)*(k-1)+val


    
def E_bound(alpha,sigma,m,q,log_M_exact_2j_array):
    val=0
    
    for j in range(m+1):
        if not(alpha==int(alpha) and alpha<j): #otherwise this j doesn't contribure
            sum1=0.
            for ell in range(j):                
                sum1=sum1+np.log(np.abs(alpha-ell))
            sum2=0.
            for ell in range(m-j):
                sum2=sum2+np.log(alpha+ell-1)
                
            
            if alpha-j<=0:                                   
                I=np.log(1-q)*(alpha-j)+log_B_bound_precomputed(m,log_M_exact_2j_array)
            
            else:
                val2=B_bound_precomputed(m,log_M_exact_2j_array)
                alpha_ceil=np.ceil(alpha)
                for ell in range(int(alpha_ceil)-j+1):
                    D=0
                    for ell2 in range(ell):
                        D=D+np.log((alpha_ceil-j-ell2)/(m+ell-ell2))
                    
                    val2=val2+np.exp(ell*np.log(q)+D+log_B_bound_precomputed(ell+m,log_M_exact_2j_array))
                
                I=np.log(val2)
            
            val=val+np.exp(np.log(q)*m-log_factorial(m)-(alpha+m-j-1)*np.log(1-q)+log_comb(m,j)+sum1+sum2+I)
        
        
    return val


def FS_RDP_woR_ro(alpha,sigma,m,q):
    #precompute required M_exact values for speed
    if (int(np.ceil(alpha))+m)%2==0:
        n_precompute=int((m+np.ceil(alpha))/2)+1
    else:
        n_precompute=int((m+np.ceil(alpha)+1)/2)+1

    n_precompute=int(n_precompute)    

    log_M_exact_2j_array=np.zeros(n_precompute)
    for j in range(n_precompute):
        log_M_exact_2j_array[j]=log_M_exact(sigma,2*j)
   
    
    expansion_terms=1.+q**2*alpha*(alpha-1)*(np.exp(4./sigma**2)-np.exp(2./sigma**2))
    for k in range(3,m):
        expansion_terms=expansion_terms+np.exp(np.log(q)*k-log_factorial(k)+log_F_deriv_bound(alpha,sigma,k,log_M_exact_2j_array))


    
    return 1./(alpha-1)*np.log(expansion_terms+E_bound(alpha,sigma,m,q,log_M_exact_2j_array))

#One step RDP bound for Poisson subsampling under replace-one adjacency (Theorem C.9)
def Poisson_RDP_ro(alpha,sigma,m,q):
    #precompute required M_exact values for speed
    if (int(np.ceil(alpha))+m)%2==0:
        n_precompute=int((m+np.ceil(alpha))/2)+1
    else:
        n_precompute=int((m+np.ceil(alpha)+1)/2)+1

    n_precompute=int(n_precompute)    

    log_M_exact_2j_array=np.zeros(n_precompute)
    for j in range(n_precompute):
        log_M_exact_2j_array[j]=log_M_exact(2*sigma,2*j)
   
    
    expansion_terms=1.+q**2*alpha*(alpha-1)*(np.exp(1./sigma**2)-np.exp(-1./sigma**2))
    for k in range(3,m):
        expansion_terms=expansion_terms+np.exp(np.log(q)*k-log_factorial(k)+log_F_deriv_bound(alpha,2*sigma,k,log_M_exact_2j_array))


    
    return 1./(alpha-1)*np.log(expansion_terms+E_bound(alpha,2*sigma,m,q,log_M_exact_2j_array))


##One step RDP bound for fixed-size subsampling with replacement under add/remove adjacency (Theorem 3.6)
#K is the number of terms that are not bounded using the worst-case result
def FS_RDP_wR_ar(alpha,sigma,m,B,D,K=None):
    if K is None:
        K=np.minimum(m-1,B)
        
    #compute q_tilde
    q_tilde=0.
    for n in range(1,K+1):
        q_tilde=q_tilde+np.exp(log_comb(B,n)+np.log(D)*(-n)+np.log(1.-1./D)*(B-n))
    q_tilde=1./(1.+(1.-1./D)**B/q_tilde)

    exponents=[0.]
    for n in range(1,K+1):
        log_a_n= log_comb(B,n)+np.log(D)*(-n)+np.log(1.-1./D)*(B-n)
        log_a_n_tilde=log_a_n-np.log(q_tilde)
        tmp=log_H_bound_minus_1(alpha,sigma/n,m,q_tilde)
        #take minimum of Taylor expansion bound and worst-case bound of each term
        tmp=np.minimum(tmp,np.log(1.-np.exp(-2*alpha*(alpha-1)*n**2/sigma**2))+2*alpha*(alpha-1)*n**2/sigma**2)
        exponents.append(log_a_n_tilde+tmp)
    for n in range(K+1,B+1):
        log_a_n= log_comb(B,n)+np.log(D)*(-n)+np.log(1.-1./D)*(B-n)
        tmp=np.log(1.-np.exp(-2*alpha*(alpha-1)*n**2/sigma**2))+2*alpha*(alpha-1)*n**2/sigma**2
        exponents.append(log_a_n+tmp)

    exp_max=np.max(exponents)
    

    
    return 1./(alpha-1.)*(np.log(np.sum(np.exp(exponents-exp_max)))+exp_max)




#Lower bound on FS-RDP with replacement from Theorem 3.7 and Appendix D.2.1.
    
#lb computed using the relaxation from Appendix D.2.1: indices_function(k) should return indices of terms used in stage k of the recursion
def log_FS_wR_ar_LB_inductive_approx(k,c,d,B,D, indices_function):
    if k==2:
        log_F_terms=[]
        indices_k=indices_function(k)
        for n in indices_k:
            log_a_n=log_comb(B,n)-n*np.log(D)+(B-n)*np.log(1-1/D)
            
            log_F_terms.append(log_a_n+d*n+B*np.log((1-1/D)*np.exp(-(c*n+d))+D**(-1))+B*(c*n+d))
       
        max_log_F=np.max(log_F_terms)    
        F_terms=np.exp(log_F_terms-max_log_F)

        return np.log(np.sum(F_terms))+max_log_F
    else:
        log_F_terms=[]
        indices_k=indices_function(k)
        for n in indices_k:
            log_F_terms.append(log_comb(B,n)+np.log(D)*(-n)+np.log(1-1/D)*(B-n)+d*n+log_FS_wR_ar_LB_inductive_approx(k-1,c,d+c*n,B,D,indices_function))
        max_log_F=np.max(log_F_terms)    
        
        F_terms=np.exp(log_F_terms-max_log_F)

        return np.log(np.sum(F_terms))+max_log_F
    
#FSR lower bound computed using selected indices
def FS_RDP_wR_ar_LB_approximate(alpha,sigma,B,D,indices_function):
    return 1/(alpha-1)*log_FS_wR_ar_LB_inductive_approx(alpha,4/sigma**2,0,B,D,indices_function)


#exact formula for the Theorem 3.7 lower bound on FS-RDP with replacement  (don't use if unless |B| and alpha are small!)
#exact lb for alpha=2
def FS_wR_ar_LB_2_exact(c,d,B,D):
    log_F_terms=[]
   
    for n in range(B+1):
        log_a_n=log_comb(B,n)-n*np.log(D)+(B-n)*np.log(1-1/D)
        
        log_F_terms.append(log_a_n+d*n+B*np.log((1-1/D)*np.exp(-(c*n+d))+D**(-1))+B*(c*n+d))
   
    max_log_F=np.max(log_F_terms)    
    F_terms=np.exp(log_F_terms-max_log_F)

    return np.log(np.sum(F_terms))+max_log_F

def FS_wR_ar_LB_inductive_exact(k,c,d,B,D):
    if k==2:
        return np.exp(FS_wR_ar_LB_2_exact(c,d,B,D))
    else:
        F=0
        for n in range(B+1):
            F=F+comb(B, n, exact=True)*D**(-n)*(1-1/D)**(B-n)*np.exp(d*n)*FS_wR_ar_LB_inductive_exact(k-1,c,d+c*n,B,D)
        return F

def FS_RDP_wR_ar_LB_exact(alpha,sigma,B,D):
    return 1/(alpha-1)*np.log(FS_wR_ar_LB_inductive_exact(alpha,4/sigma**2,0,B,D))

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


N_alpha=500

alpha_array=1+10**np.linspace(-1,1.5,N_alpha)

m_array=[4]
N_m=len(m_array)

deltas = [1e-5, 1e-6, 1e-7, 1e-8,1e-9, 1e-10,1e-11,1e-12]
n_epochs = 250
n_train = 50000
batch_size = 120
accu_factor = n_train / batch_size * n_epochs

eps_array=np.zeros((N_m,N_alpha))


for j1 in range(N_m):
    for j2 in range(N_alpha):
        eps_array[j1,j2]=FS_RDP_woR_ro(alpha_array[j2], sigma, m_array[j1], q)

eps_array_poisson = np.zeros((N_m,N_alpha))

for j1 in range(N_m):
    for j2 in range(N_alpha):
        eps_array_poisson[j1,j2]=Poisson_RDP_ro(alpha_array[j2], sigma, m_array[j1], q)


Wang_eps_array=np.zeros(N_alpha)
for j1 in range(N_alpha):
    Wang_eps_array[j1]=Wang_et_al_upper_bound(alpha_array[j1])
    
    
alpha_lb_array=[]
Wang_eps_lb_array=[]
for alpha in range(2,int(max(alpha_array))+1):
    alpha_lb_array.append(alpha)
    Wang_eps_lb_array.append(Wang_et_al_lower_bound(alpha))


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
