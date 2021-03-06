


=============================================================================


     * AutoNOM (Automatic Nonstationary Oscillatory Modelling) *


           Julia scripts which implement the methodology for

 "Bayesian Model Search for Nonstationary Periodic Time Series" 
                (2019), Hadj-Amar et al.

	                    version 0.5

==============================================================================



We provide Julia scripts to model nonstationary
periodic time series. The methods implement reversible-jump MCMC
algorithms based on he paper of Hadj-Amar et al. (2019). 


 ------  Note: ------

	This version does not use the expensive optimization used to
	sample the linear basis function coefficients \beta's. We now draw
	the \beta's directly from their Gaussian posterior conditional 
	distribution.

 -------------------




========================================================================
Summary of AutoNOM package contents:
========================================================================


* Main inference script:

 illustrative_example.jl


* Functions and utilities: 

 functions_stationary_model.jl
 functions_non_stationary_model.jl


========================================================================
Setup and Usage Examples
========================================================================


To use AutoNOM in illustrative_example.jl you must first add 
add the directory that contains functions_stationary_model.jl
and functions_stationary_model.jl to path. The example is based
on the Illustrative Example of Section 5.1 in the manuscript, apart
that this version does not use the expensive optimization used to
sample the linear basis function coefficients \beta's. We now draw
the \beta's directly from their Gaussian posterior conditional 
distribution.



========================================================================
Contact Information 
========================================================================

Beniamino Hadj-Amar: B.Hadj-Amar@warwick.ac.uk

https://warwick.ac.uk/fac/sci/statistics/staff/research_students/hadjamar/

========================================================================
Data
========================================================================
We also provide four dataset: (1 skin temperature and 3 breathing traces of a rat)





Permission is granted for anyone to copy, use, or modify these
programs and accompanying documents for purposes of research or
education.  As the programs were written 
for research purposes only, they have not been tested so
that would be advisable in any important application.  
All use of these programs is entirely at the user's own risk.
	


