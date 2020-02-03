
# AutoNOM - Automatic Nonstationary Oscillatory Modelling 

This software package includes Julia (v0.62) scripts to model nonstationary
periodic time series. The methods implement reversible-jump MCMC based
algorithms as described in the paper of Hadj-Amar et al. (2019), 
"Bayesian Model Search for Nonstationary Periodic Time Series".

Note that this version does not use the optimization routine used to
sample the linear basis function coefficients \beta's (normal approximation).
which are now sampled from their Gaussian posterior conditional 
distribution. For more details, see Ph.D. Thesis of B.Hadj-Amar. 


* Main Inference Script:

 illustrative_example.jl


* Functions and Utilities: 

 functions_stationary_model.jl, functions_non_stationary_model.jl
 

