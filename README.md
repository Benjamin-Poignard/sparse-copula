# Sparse copula models

Matlab implementation of sparse copula models with pseudo-observations replicating the results of the paper:

*Sparse M-estimators in semi-parametric copula models* by Jean-David Fermanian and Benjamin Poignard.

Link: https://arxiv.org/abs/2112.12351


# Overview

The code in this replication package reproduces:

- The simulation results provided in Table 1, Section 5.2: the replicator should execute program *simulations_sparse_gaussian_copula.m*.  

- The simulation results provided in Table 2, Section 5.2: the replicator should execute program *simulations_sparse_conditional.m*.  

- The simulation experiment displayed in Figure 1, Appendix: the replicator should execute program *simulations_sensitivity_sparse_gaussian_copula.m*.  

# Software requirements

The Matlab code was run on a Mac-OS Apple M1 Ultra with 20 cores and 128 GB Memory. The version of the Matlab software on which the code was run is a follows: 9.12.0.1975300 (R2022a) Update 3. 

The following toolboxes should be installed:

- Statistics and Machine Learning Toolbox, Version 12.3.
- Parallel Computing Toolbox, Version 7.6.
- Global Optimization Toolbox, Version 4.7.
- Optimization Toolbox, Version 9.3.

Parallel Computing Toolbox is highly recommended to run the code to speed up the cross-validation procedure employed to select the optimal tuning parameter. All the run-time requirements displayed below are reported when the code is run with the Parallel Computing Toolbox.

# Controlled randomness

For the sake of replication, seeds have been specified in *simulations_sparse_gaussian_copula.m*, *simulations_sparse_conditional.m* and *simulations_sensitivity_sparse_gaussian_copula.m*. To be specific:
- in *simulations_sparse_gaussian_copula.m*, seeds are set in: lines 3, 176, 349 and 517.
- in *simulations_sparse_conditional.m*, seeds are set in: lines 3, 95, 195 and 286.
- in *simulations_sensitivity_sparse_gaussian_copula.m*, the seed is set in line 2.

# Run-time Requirements

The files for replicating the simulated experiments are split into sections according to the dimension, copula and sample sizes:

- *simulations_sparse_gaussian_copula.m* contains four sections: for each dimension problem (p=10, p=20), two sample sizes (n=500, n=1000).

- *simulations_sparse_conditional.m* contains four sections: for each copula model (Gumbel and Clayton), two sample sizes (n=500, n=1000).

The approximate time needed to reproduce the simulated experiments on a Mac-OS M1 Ultra desktop machine is as follows:

- To run each section of the file *simulations_sparse_gaussian_copula.m*, the approximate computation time is: 8 hours when p = 10 and 24 hours when p = 20. 
- To run file *sensitivity.m*: approximately 120 hours.
- To run each section of the file *simulations_sparse_conditional.m*, the approximate computation time is: 70 hours (resp. 122 hours) for the Gumbel copula when n = 500 (resp. n = 1000); 37 hours (resp. 46 hours) for the Clayton copula when n = 500 (resp. n = 1000).

**Remark:** The computation time highly depends on the grid size selected for cross-validation to choose the optimal penalization parameter "\lambda_n" (called tuning or regularization parameter): in all the simulated experiments, the optimal tuning parameter is searched in the grid $\{c\sqrt{\log(dim)/n}, c=0.01, 0.05, 0.1, ..., 4.5\}$, so that there are $91$ different candidates for the tuning parameter.
To save run-time computation, the user may set a smaller grid.

# Description of the code for replication

The replicator should execute the following Matlab files to replicate Table 1, Table 2 and Figure 1 of the paper.

**A. Replication of the results reported in Table 1**

- Program *simulations_sparse_gaussian_copula.m* will replicate the results reported in Table 1. 
  - "Results_1" (line 160) provides the results of Table 1 for p = 10, n = 500.
  - "Results_2" (line 328) provides the resluts of Table 1 for p = 10, n = 1000.
  - "Results_3" (line 501) provides the results of Table 1 for p = 20, n = 500.
  - "Results_4" (line 670) provides the resluts of Table 1 for p = 20, n = 1000.

**B. Replication of the results reported in Table 2**

- Program *simulations_sparse_conditional.m* will replicate the results reported in Table 2:
  - "Results_1" (line 92) provides the results of Table 2 for the Gumbel copula, n = 500.
  - "Results_2" (line 182) provides the resluts of Table 2 for the Gumbel copula, n = 1000.
  - "Results_3" (line 283) provides the results of Table 2 for the Clayton copula, n = 500.
  - "Results_4" (line 374) provides the resluts of Table 2 for the Clayton copula, n = 1000.

**C. Replication of Figure 1**

- Program *sensitivity.m* will replicate Figure 1: the sensitiviy patterns for each loss function (Gaussian, least squares) and penalty function (SCAD, MCP) are saved in 'Metrics.mat'.
