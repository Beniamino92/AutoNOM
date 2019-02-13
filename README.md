# AutoNOM
Auomatic Nonstationary Oscillatory Modelling

This software package includes Julia scripts to model nonstationary
periodic time series. The methods implement reversible-jump MCMC based
algorithms as described in the paper of Hadj-Amar et al. (2019), 
"Bayesian Model Search for Nonstationary Periodic Time Series".


* Main inference script:

 illustrative_example.jl


* Functions and utilities: 

 functions_stationary_model.jl
 
 functions_non_stationary_model.jl




To use AutoNOM in illustrative_example.jl you must first add 
add the directory that contains functions_stationary_model.jl
and functions_stationary_model.jl to path. The example is based
on the Illustrative Example of Section 5.1 in the manuscript. 


* Contact Information 


Beniamino Hadj-Amar: B.Hadj-Amar@warwick.ac.uk

https://warwick.ac.uk/fac/sci/statistics/staff/research_students/hadjamar/


