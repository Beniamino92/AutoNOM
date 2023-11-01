
# AutoNOM - Automatic Nonstationary Oscillatory Modelling 

`AutoNOM` is a Julia software to model nonstationary
periodic time series. This Bayesian method approximates the time series using a piecewise oscillatory model with unknown periodicities, to estimate the change-points while simultaneously identifying the potentially changing periodicities in the data. Inference is carried out through reversible-jump MCMC based
algorithms as detailed in Hadj-Amar et al. (2020) ["_Bayesian Model Search for Nonstationary Periodic Time Series_"]([https://www.cell.com/current-biology/fulltext/S0960-9822(23)01355-6](https://www.tandfonline.com/doi/full/10.1080/01621459.2019.1623043)), published in JASA. The software is implemented in ``Julia 1.6``.




## Example 

We provide a snapshot of `illustrative_example.jl` for using our software in Julia

* Run MCMC sampler
  ```Julia
  MCMC_simul = AutoNOM(data, hyperparms; s_start = [40])
  ```
* Plot estimated signal + simulated data
<p align="center">
<img src="https://github.com/Beniamino92/AutoNOM/blob/master/figures/posterior_data.png" width="400" heigth="140"/> 
</p>

* Trace plot frequencies 
<p align="center">
<img src="https://github.com/Beniamino92/AutoNOM/blob/master/figures/posterior_freq.png" width="400" heigth="170"/> 
</p>


 

