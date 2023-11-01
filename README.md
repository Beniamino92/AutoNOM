
# AutoNOM - Automatic Nonstationary Oscillatory Modelling 

`AutoNOM` is a Julia software to model nonstationary
periodic time series. This Bayesian method approximates the time series using a piecewise oscillatory model with unknown periodicities, to estimate the change-points while simultaneously identifying the potentially changing periodicities in the data. Inference is carried out through reversible-jump MCMC based
algorithms as detailed in Hadj-Amar et al. (2020) ["_Bayesian Model Search for Nonstationary Periodic Time Series_"]([https://www.cell.com/current-biology/fulltext/S0960-9822(23)01355-6](https://www.tandfonline.com/doi/full/10.1080/01621459.2019.1623043)), published in JASA.


## Example 

We provide a snapshot of `illustrative_example.jl` for using our software in Julia

* Run MCMC sampler
  ```Julia
  MCMC_simul = AutoNOM(data, hyperparms; s_start = [40])
  ```

* Estimated frequency in each segment
```Julia
segment = 1
p = plot(layout = (n_freq_est[segment], 1))
for l in 1:n_freq_est[segment]
       index_CP_nfreq_est = findall(n_freq_final[segment, :,  :][index_CP_est] .== n_freq_final[segment])
       plot!(p[l,1], ω_final[segment, l, index_CP_nfreq_est])
       hline!(p[l, 1], [ω_true[segment][l]], color = :red, linewidth = 2, linestyle = :dot, legend = nothing)
end 
display(p)
```
 

