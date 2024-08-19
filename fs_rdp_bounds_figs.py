#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###Test of fixed-size subsampling (with and without replacement) Renyi Differential Privacy (FS-RDP) methods from the paper
##Renyi DP-SGD with Fixed-Size Minibatches: Tighter Guarantees with or without Replacement 

import math
import numpy as np
import matplotlib.pyplot as plt
from FS_RDP_bounds import FS_RDP_woR, FS_RDP_woR_convexity_method, FS_RDP_wR, FS_RDP_wR_LB_approximate, FS_RDP_wR_LB_exact


##Implementation of bounds from the paper Renyi differential privacy and analytical moments accountant by Wang et al, for comparison purposes
def K_Wang_et_al(alpha,sigma,q):
    K_terms=[1.]
    alpha_prod=alpha*(alpha-1)
    
    K_terms.append(2*alpha_prod*q**2*(np.exp(4/sigma**2)-1))
    
    for j in range(3,int(alpha+1)):
        alpha_prod=alpha_prod*(alpha-j+1)
        K_terms.append(2*q**j*alpha_prod/np.math.factorial(j)*np.exp((j-1)*2*j/sigma**2))

    #Sum terms in reverse order for numerical stability
    K=0
    for j in range(len(K_terms)):
        K=K+K_terms[len(K_terms)-1-j] 
    K=np.log(K)
    return K

def Wang_et_al_upper_bound(alpha,sigma,q):
    if alpha>=2:
        if int(alpha)==alpha:
            return 1./(alpha-1.)*K_Wang_et_al(alpha,sigma,q)
        else:
            return (1.-(alpha-math.floor(alpha)))/(alpha-1)*K_Wang_et_al(math.floor(alpha),sigma,q)+(alpha-math.floor(alpha))/(alpha-1)*K_Wang_et_al(math.floor(alpha)+1,sigma,q)
    else:
        return Wang_et_al_upper_bound(2,sigma,q)

def Wang_et_al_lower_bound(alpha,sigma,q):
    if int(alpha)==alpha:
        L_terms=[1.]
        L_terms.append(alpha*q/(1-q))
        alpha_prod=alpha
        for j in range(2,alpha+1):
            alpha_prod=alpha_prod*(alpha-j+1)
            L_terms.append(alpha_prod/np.math.factorial(j)*(q/(1-q))**j*np.exp((j-1)*2*j/sigma**2))
        
        #Sum terms in reverse order for numerical stability
        L=0
        for j in range(len(L_terms)):
            L=L+L_terms[len(L_terms)-1-j]         
        return alpha/(alpha-1)*np.log(1-q)+1/(alpha-1)*np.log(L)
    else:
        print("Error, alpha must be an integer.")
        
        
##########Comparison of our FS-RDP without replacement upper bound with results from Wang et al
# #make plots: large alpha 
N_alpha=500
alpha_array=1+10**np.linspace(-2,1.5,N_alpha)
m_array=[3,4,6,8,10]

B=120
D=50000
N_epochs=250

sigma=6.



q=B/D
N_m=len(m_array)


FS_eps_array=np.zeros((N_m,N_alpha))

for j1 in range(N_m):
    for j2 in range(N_alpha):
        FS_eps_array[j1,j2]=FS_RDP_woR(alpha_array[j2],sigma,m_array[j1],q)


integer_alpha_method_array=np.zeros(N_alpha)       
Wang_eps_array=np.zeros(N_alpha)
for j1 in range(N_alpha):
    integer_alpha_method_array[j1]=FS_RDP_woR_convexity_method(alpha_array[j1],sigma,q)
    Wang_eps_array[j1]=Wang_et_al_upper_bound(alpha_array[j1],sigma,q)
    
    
alpha_lb_array=[]
Wang_eps_lb_array=[]
for alpha in range(2,int(max(alpha_array))+1):
    alpha_lb_array.append(alpha)
    Wang_eps_lb_array.append(Wang_et_al_lower_bound(alpha,sigma,q))

#N_iters=math.floor(D/B*N_epochs)
N_iters=1.



plt.figure()
labels=[]
for k in range(N_m):
    plt.semilogy(alpha_array,N_iters*FS_eps_array[k,:])
    labels.append("Thm.3.3 upper bound (m = {})".format(m_array[k]))
    
m_idx=1
# plt.semilogy(alpha_array,N_iters*np.minimum(integer_alpha_method_array,FS_eps_array[m_idx,:]),'-',color='0')
# labels.append("Thm.3.3 and convexity technique")

plt.semilogy(alpha_array,N_iters*Wang_eps_array,'-.')
labels.append("Wang et al. upper bound")
plt.semilogy(alpha_lb_array,N_iters*np.array(Wang_eps_lb_array),'.')
labels.append("Wang et al. lower bound")

plt.legend(labels)
plt.xlabel(r'$\alpha$')
plt.ylabel('One-step RDP Bound')


plt.savefig("comp_with_Wang_et_al_no_convexity.pdf")


#make plots: small alpha
N_alpha=500
alpha_array=1+10**np.linspace(-2,.5,N_alpha)
m_array=[4]

B=120
D=50000
N_epochs=250

sigma=6.



q=B/D
N_m=len(m_array)



FS_eps_array=np.zeros((N_m,N_alpha))

for j1 in range(N_m):
    for j2 in range(N_alpha):
        FS_eps_array[j1,j2]=FS_RDP_woR(alpha_array[j2],sigma,m_array[j1],q)
        

integer_alpha_method_array=np.zeros(N_alpha)       
Wang_eps_array=np.zeros(N_alpha)
for j1 in range(N_alpha):
    integer_alpha_method_array[j1]=FS_RDP_woR_convexity_method(alpha_array[j1],sigma,q)
    Wang_eps_array[j1]=Wang_et_al_upper_bound(alpha_array[j1],sigma,q)
    
    
alpha_lb_array=[]
Wang_eps_lb_array=[]
for alpha in range(2,int(max(alpha_array))+1):
    alpha_lb_array.append(alpha)
    Wang_eps_lb_array.append(Wang_et_al_lower_bound(alpha,sigma,q))
    
    
# N_iters=math.floor(D/B*N_epochs)
N_iters=1

plt.figure()
labels=[]
plt.plot(alpha_array,N_iters*integer_alpha_method_array,'-')
labels.append("Thm.3.1 upper bound with convexity technique")

for k in range(N_m):
    plt.plot(alpha_array,N_iters*FS_eps_array[k,:])
    labels.append("Thm.3.3 upper bound (m = {})".format(m_array[k]))
    


plt.plot(alpha_lb_array,N_iters*np.array(Wang_eps_lb_array),'o')
labels.append("Wang et al. lower bound")

plt.legend(labels)
plt.xlabel(r'$\alpha$')
plt.ylabel('One-step RDP Bound')

plt.savefig("comp_with_Wang_et_al_small_alpha.pdf")





##########Exploration of our FS-RDP with replacement upper and lower bounds
# #make plots: small alpha 
alpha_array=[j for j in range(2,11)]
for alpha in np.linspace(1.001,10,500):
    alpha_array.append(alpha)
alpha_array=np.sort(np.array(alpha_array))
N_alpha=len(alpha_array)

m=8

FSR_eps_array=np.zeros(N_alpha)
FS_eps_array=np.zeros(N_alpha)
for j in range(N_alpha):
    FSR_eps_array[j]=FS_RDP_wR(alpha_array[j],sigma,m,B,D)
    FS_eps_array[j]=FS_RDP_woR(alpha_array[j],sigma,m,q)
        


alpha_lb_array=np.array(range(2,10+1))

indices=[B]
FSR_eps_lb_general_array=np.zeros(len(alpha_lb_array))

for j in range(len(alpha_lb_array)):
    alpha=alpha_lb_array[j]
    

    FSR_eps_lb_general_array[j]=FS_RDP_wR_LB_approximate(alpha,sigma,B,D,lambda k: indices if k >2 else np.array(range(0,B+1)))


 


plt.figure()
labels=[]


plt.semilogy(alpha_array,N_iters*FSR_eps_array,color='black')
labels.append('$FS_{wR}$-RDP upper bound')
    
plt.semilogy(alpha_lb_array,N_iters*np.array(FSR_eps_lb_general_array),'o')
if len(indices)>1:
    labels.append(r'$FS_{wR}$-RDP lower bound ('+str(len(indices))+' terms)')
else:
    labels.append(r'$FS_{wR}$-RDP lower bound ($n=|B|$ term)')

plt.semilogy(alpha_array,N_iters*np.array(FS_eps_array),'--')
labels.append('$FS_{woR}$-RDP upper bound')


plt.legend(labels)
plt.xlabel(r'$\alpha$')
plt.ylabel('One-step RDP Bound')
plt.savefig("FSR_upper_lower_bounds.pdf")



    
#exact one step FS_{wR}-RDP bound as a function of B for fixed q

N_sigma=5
sigma_array=np.linspace(2,6,N_sigma)

B_i=2
B_f=200
B_array=[]

alpha=2
q=.001

FSR_eps_B_array=np.zeros((B_f+1-B_i,N_sigma))
for B in range (B_i,B_f+1):
    print(B)
    B_array.append(B)
    D=B/q
    for j in range(N_sigma):

        FSR_eps_B_array[B-B_i,j]=FS_RDP_wR_LB_exact(alpha,sigma_array[j],B,D)
    
plt.figure()
labels=[]
for k in range(N_sigma):
    plt.semilogy(B_array,FSR_eps_B_array[:,k])
    labels.append(r"$\sigma={}$".format(sigma_array[k]))
    
plt.legend(labels, loc='center right')
plt.xlabel(r'$|B|$')
plt.ylabel('One-step RDP Lower Bound')
    
plt.savefig("FSR_exact_vs_B.pdf")


#lower bound as a function of alpha for different |B|'s
sigma=6.
q=.001


# B_array=[4]#,20,30,40,60,80]
B_array=[20,30,40,60,80]
alpha_lb_array=np.array(range(2,10+1))

eps_lb_array=np.zeros((len(alpha_lb_array),len(B_array)))

for j1 in range(len(B_array)):
    B=B_array[j1]
    D=B/q
        
    for j2 in range(len(alpha_lb_array)):
        alpha=alpha_lb_array[j2]
        
        if alpha>8:
            def lb_indices(k):
                if k==2:
                    return np.array(range(0,B+1))
                else:
                    return [0,1,2,B]
        elif alpha==7:
            def lb_indices(k):
                if k==2:
                    return np.array(range(0,B+1))
                else:
                    return [0,1,2,3,B-1,B]
        elif alpha==6:
            def lb_indices(k):
                if k==2:
                    return np.array(range(0,B+1))
                else:
                    return [0,1,2,3,B-2,B-1,B]
        elif alpha<5:
            def lb_indices(k):
                if k==2:
                    return np.array(range(0,B+1))
                else:
                    return [0,1,2,3,4,5,B-3,B-2,B-1,B]
           
                
        print("B={}, alpha={}".format(B,alpha))

        eps_lb_array[j2,j1]=FS_RDP_wR_LB_approximate(alpha,sigma,B,D,lb_indices)

    
plt.figure()
labels=[]
for k in range(len(B_array)):
    plt.semilogy(alpha_lb_array,eps_lb_array[:,k])
    labels.append(r"$|B|={}$".format(B_array[k]))
    
plt.legend(labels,loc='lower right')
plt.xlabel(r'$\alpha$')
plt.ylabel('One-step RDP Lower Bound')
    
plt.savefig("FSR_LB_vs_alpha.pdf")

        
Wang_LB_array=np.zeros(len(alpha_lb_array))
for j in range(len(alpha_lb_array)):
    Wang_LB_array[j]=Wang_et_al_lower_bound(alpha_lb_array[j2],sigma,q)




