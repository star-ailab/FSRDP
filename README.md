# Differentially Private Stochastic Gradient Descent with Fixed-Size Minibatches: Tighter RDP Guarantees with or without Replacement

# Requirements:

- Python 3.8.0
- PyTorch 1.13.1
- Matplotlib

# Usage:

## To obtain the bounds in our accountant run:
    - python fs_rdp_bounds.py

This python file can be used in other applications to utilize our accountant. It contains functions for subsampled mechanisms with and without replacement as well as different implementations for replace-one and add/remove relationship. Below we provide a usage example:

## CIFAR-10 example: To privately train a model with our accountant on CIFAR-10 run:

    - python fs_rdp_accountant_cifar_example

A pretrained model is saved under the folder named "saved_models." This is the same as the one used in Abadi et al to non-privately pretrain on CIFAR-100. It can be replaced by other deep learning models.
In order to non-privately train this model on CIFAR-100 locally, one can set the --pretrain switch to "store_false," and then run the above command.

## To generate Figure 1 and all other figures in the Appendices run:

    - python fs_rdp_bounds_figs.py

## To generate Figure 2 and Figure 3 in the paper run:

    - python fsrdP_accountant_figs_2_3

## To generate Figure 4 in the paper run:

    - python eps_vs_delta_cifar_fig_4:
