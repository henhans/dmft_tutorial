# Tutorial for dynamical mean-field theory using exact-diagonalization impurity solver

This is a dynamical mean-field theory (DMFT) tutorial for [2024 Strongly correlated physics – Numerical and Analytical approaches](https://www.phys.ncts.ntu.edu.tw/act/actnews/2024-Strongly-correlated-physics-%E2%80%93-Numerical-and-Analytical-approaches-44671449/home "2024 Strongly correlated physics – Numerical and Analytical approaches") at National Yang-Ming Chiao-Tung University and National Center of Theoretical Science, Taiwan. In this tutorial, you will learn how to implement the DMFT algorithm using exact-diagonalization (ED) as an impurity solver. The repository contains three folders.

- **metal**: This folder contains a jupyter notebook ```dmft-ed-metal.ipynb``` implementing the DMFT loop to investigate the correlated metal behavior. The code shows the behavior of the Green's function and self-energy on the real and Matsubara frequencies. The script ```dos.py``` contains a simple semicircular density of states of the Bethe lattice. The script ```ed.py``` is a simple but less efficient implementation of the exact-diagonalization algorithm for solving the Anderson impurity model.

- **mott**: This folder is similar to the metal folder, but it applies the DMFT to the Mott insulating solution and shows the behavior of the Green's function and self-energy on the real frequency and Matsubara frequency.

- **metal_mott_transition**: This folder applies the DMFT to investigate the first-order metal-Mott-insulator transition at half-filling. In order to investigate the first-order transition, we run the DMFT calculations from small Coulomb interaction U to large U (the metal2mott folder with ```dmft-ed-metal2mott.ipynb```) as well as from large U to small U (the mott2metal folder with ```dmft-ed-mott2metal.ipynb```). Then, we plot the hysteresis curves of the first order transition (in ```dmft-ed-metal2mott.ipynb```) showing in the double-occupancies vs. U and the quasiparticle weight Z vs. U figures. The density of states as a function of U is also plotted.

- **Optional homework**: Can you reproduce the finite temperature phase diagram of the DMFT metal-insulator transition using the provided code?

## References
<a id="1">[1]</a> 
Dynamical mean-field theory of strongly correlated fermion systems and the limit of infinite dimensions,
A Georges, G Kotliar, W Krauth, MJ Rozenberg,
Reviews of Modern Physics 68 (1), 13 (1996).
