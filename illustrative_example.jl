# ------------------------------------------------------------------------

       # -- AutoNOM (Automatic Nonstationary Oscillatory Modelling) -- #

# ------------------------------------------------------------------------


using StatsBase
using Distributions
using Optim
using DSP
using ProgressMeter
using Plots
using Random
using LinearAlgebra
using FreqTables

# selected folder must contain '/include'
path_AutoNOM = "/Users/beniamino/Desktop_New/Research/Autonom_NEW/"
cd(path_AutoNOM) 

include(pwd()*"/include/util_stationary.jl")
include(pwd()*"/include/util_nonstationary.jl")
include(pwd()*"/include/autoNOM.jl")





# --------------------------------------------------------------------------- #
# ----------------- Simulation Study, Illustrative Example ------------------ #
# --------------------------------------------------------------------------- #





standardize = false # (whether time series should be standardized or not )
n_obs = 900 # n of observations
s_true = [300, 650] # change-points locations

# Frequencies, for each segment.
ω_1 = [1/24, 1/15, 1/7]
ω_2 = [1/18]
ω_3 = [1/22, 1/15]
ω_true = [ω_1, ω_2, ω_3]

# Linear basis coefficients, for each segment.
β_1 = [2.0, 3.0, 4.0, 5.0, 1.0, 2.5]
β_2 = [4.0, 3.0]
β_3 = [2.5, 4.0, 4.0, 2.0]
β_true = [β_1, β_2, β_3]

# Intercept, Linear trend and Residual error, for each segment.
α_true = [0.0, 0.0, 0.0]
μ_true = [0.01,  0.0, -0.005]
σ_true = [4.0, 3.5, 2.8]

# True values, all : 
parms_true = Dict()
parms_true[:ω] = ω_true
parms_true[:β] = β_true 
parms_true[:s] = s_true
parms_true[:α] = α_true 
parms_true[:μ] = μ_true 
parms_true[:σ] = σ_true 

# ------------- Simulation Time Series  ------------

Random.seed!(108)

# Time Series
illustrative_example = get_data(n_obs, parms_true)
data = illustrative_example[:data]
signal = illustrative_example[:signal]
if (standardize) 
     data = (data .- mean(data))./std(data) 
     signal = (signal .- mean(signal))./std(signal) 
end 


# Plot: Observations and Signal 
plot(1:n_obs, data, legend = nothing)
plot!(1:n_obs, signal, color = :red)
vline!(s_true, color = :black, linewidth = 2, linestyle = :dot)



# ------------ Parameters MCMC for AutoNOM ----------

hyperparms = Dict()
hyperparms[:n_iter_MCMC] = 20000 # number of MCMC iteration for AutoNOM
hyperparms[:n_CP_max] = 10 # maximum number of change-points
hyperparms[:n_freq_max] = 8 # maximum number of frequencies in each segment. 
hyperparms[:σ_β] = 10 # prior variance for β,  β ∼ Normal(0, σ_β * I)
hyperparms[:ν0] = 1/100 # prior σ², InverseGamma(ν0/2, η0/2)
hyperparms[:γ0] = 1/100 # prior σ², InverseGamma(ν0/2, η0/2)
hyperparms[:λ_S] = 1 # poisson prior parameter, i.e n of freq ∼ Poisson(λ_S)
hyperparms[:δ_ω_mixing_brn] = 0.2  # mixing probability random walk when relocation a frequency
#                        (after 300 iterations burn-in)
hyperparms[:c_S] = 0.4 # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
hyperparms[:ϕ_ω] = 0.25 # birth step, frequency is sampled from Unif(0, ψ_ω)
hyperparms[:ψ_ω] = 10/n_obs # miminum distance between frequencies
hyperparms[:σ_RW_ω_hyper] = 20 # variance parameter for random walk when relocating a freq :
#                   ωᵖ ∼ Normal(ωᶜ, σ_ω), where σ_ω = 1/(σ_RW_ω_hyper*n)
hyperparms[:λ_NS] = 1/10 # poisson prior parameter, i.e n of cp ∼ Poisson(λ_NS)
hyperparms[:c_NS] = 0.4 # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
hyperparms[:ψ_NS] = 10 # minumum distance between change-points
hyperparms[:σ_RW_s] = 2.5 # variance parameter for random walk when relocating a change-point
hyperparms[:δ_s_mixing] = 0.2 # mixing probability for mixture proposal when relocating a change-point




##  ---- AutoNOM 
MCMC_simul = AutoNOM(data, hyperparms; s_start = [40])

# - output MCMC
n_freq_sample = MCMC_simul[:m];
ω_sample = MCMC_simul[:ω];
β_sample = MCMC_simul[:β];
σ_sample = MCMC_simul[:σ];
s_sample = MCMC_simul[:s];
log_likelik_sample = MCMC_simul[:log_likelik];
n_CP_sample = MCMC_simul[:n_CP];

# ----------------------- Diagnostic Convergence  ----------------------
n_iter_MCMC = hyperparms[:n_iter_MCMC]
burn_in_MCMC = Int64(0.4*n_iter_MCMC)
final_indexes_MCMC = burn_in_MCMC:n_iter_MCMC

# Trace log likelihood
log_likelik_sample_total = zeros(Float64, n_iter_MCMC + 1)
for t in 1:(n_iter_MCMC+1)
    n_CP = n_CP_sample[t]
    log_likelik_sample_total[t] = sum(log_likelik_sample[1:(n_CP+1), t])
end

# -- Trace plot: log likelihood, for assessing convergence
plot(log_likelik_sample_total[final_indexes_MCMC])


# -- Markov Chains after burn-in
n_CP_final = n_CP_sample[final_indexes_MCMC];
β_final = β_sample[:, :, final_indexes_MCMC];
ω_final = ω_sample[:, :, final_indexes_MCMC];
s_final = s_sample[:, final_indexes_MCMC];
σ_final = σ_sample[:, final_indexes_MCMC];
n_freq_final = n_freq_sample[:, final_indexes_MCMC];
n_final = length(n_CP_final);

# -- Trace plot: n of change-points
plot(n_CP_final)

# Estimate: n of change-points
n_CP_est = mode(n_CP_final)
index_CP_est = findall(n_CP_final .== n_CP_est)

# Estimate: n of frequencies, conditioned on n_CP_est
n_freq_est = zeros(Int64, n_CP_est+1)
for j in 1:(n_CP_est+1)
    n_freq_est[j] = mode(n_freq_final[j, index_CP_est])
end

##  n of frequencies for each segment 
for j in 1:(n_CP_est + 1)
       println("Segment ", j)
       display(freqtable(n_freq_final[j, index_CP_est])/size(n_freq_final[j, index_CP_est], 1))
end 

# Trace Plot: change-points location
p = plot(s_final[1, index_CP_est], legend = nothing, ylims=(1, n_obs))
for j in 2:n_CP_est
        plot!(p, s_final[j, index_CP_est])
end
display(p)


# Estimate: change-points location
s_est = mean(s_final[1:n_CP_est, index_CP_est], dims = 2)

# Trace Plot: frequencies in selected segment
# (conditioned on n change-points and n freq in each segment)
segment = 1
p = plot(layout = (n_freq_est[segment], 1))
for l in 1:n_freq_est[segment]
       index_CP_nfreq_est = findall(n_freq_final[segment, :,  :][index_CP_est] .== n_freq_final[segment])
       plot!(p[l,1], ω_final[segment, l, index_CP_nfreq_est])
       hline!(p[l, 1], [ω_true[segment][l]], color = :red, linewidth = 2, linestyle = :dot, legend = nothing)
end 
display(p)

# Estimate: signal, by averaging over MCMC iterations
signal_MCMC = zeros(Float64, n_final, n_obs)
@showprogress for t in 1:n_final
  n_CP = n_CP_final[t]
  signal_MCMC[t, :] = get_estimated_signal(data, s_final[1:n_CP, t],
                                    β_final[:, :, t],
                                    ω_final[:, :, t],
                                    n_freq_final[:, t])
end
signal_est = reshape(mean(signal_MCMC, dims = 1), n_obs)


# Plot: data, estimated signal, true signal and estimated change-point locations
scatter(1:n_obs, data, legend = nothing)
plot!(1:n_obs, signal_est, color = :red)
vline!(s_est, color = :black, linewidth = 2, linestyle = :dot)


