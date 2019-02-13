using StatsBase
using Distributions
using PyPlot
using Optim
using DSP
using RCall
using ProgressMeter

# Directory that includes the functions 
cd("/Users/beniamino/Desktop/AutoNOM")

path_stationary_fun = pwd()*"/functions_stationary_model.jl"
path_non_stationary_fun = pwd()*"/functions_non_stationary_model.jl"

include(path_stationary_fun)
include(path_non_stationary_fun)



"""
# --- Function: generate a nonstationary periodic time series,
                see Equation (1), Hadj-Amar et al. (2019).

     - n_obs = number of observations
     - s = change-points locations
     - ω = frequencies, for each segment
     - β = linear basis function coefficients, for each segment
     - α = intercept, for each segment
     - μ = linear trend, for each segment
     - σ = residual error, for each segment

"""

function get_data(n, s, ω, β, α, μ, σ)

    if !(length(s) + 1 == length(ω)) error("DimensionMismatch") end

    s_aux = vcat(1, s, n_obs)
    n_CP = length(s)

    signal = []
    noise = []

    for j in 1:(n_CP+1)

        a = s_aux[j]
        j == n_CP + 1 ? b = n_obs : b = s_aux[j+1] - 1

        X = get_X(ω[j], a, b)

        f = X * vcat(α[j], μ[j], β[j])
        ε = rand(Normal(0, σ[j]), length(f))

        append!(signal, f)
        append!(noise, ε)
    end

    return Dict("data" => signal + noise,
                "signal" => signal)
end



"""
# --- Function: get objects MCMC and starting values,
#

     - s = starting value location change-points

     - periodogram: if true: starting values for
                        n of frequencies and ω are selected testing the peaks
                        of the periodgram
                    else: starting values for
                       n of frequencies = 1, and ω ∼ Uniform(0, 0.5)
"""

function get_MCMC_objects(s; periodogram = false)

  # --- MCMC Objects

  β_sample = zeros(Float64, n_CP_max+1, 2*n_freq_max+2, n_iter_MCMC + 1)
  ω_sample = zeros(Float64, n_CP_max+1, n_freq_max, n_iter_MCMC + 1)
  s_sample = zeros(Float64, n_CP_max, n_iter_MCMC + 1)
  σ_sample = zeros(Float64, n_CP_max + 1, n_iter_MCMC + 1)
  n_freq_sample = zeros(Int64, n_CP_max+1, n_iter_MCMC+1)
  log_likelik_sample = zeros(Float64, n_CP_max + 1, n_iter_MCMC + 1)
  β_mean_sample = zeros(Float64, n_CP_max+1, 2*n_freq_max+2, n_iter_MCMC + 1)
  n_CP_sample = zeros(Int64, n_iter_MCMC + 1)

  # ------- Initial values -------- ##

  n_CP = length(s)

  # n_CP, n_freq, s
  n_CP_sample[1] = n_CP
  s_sample[1:n_CP_sample[1]] = s

  s_aux = Array{Int64}(vcat(1, s, n_obs))

  # β, ω, σ
  for j in 1:(n_CP_sample[1]+1)

    global a = s_aux[j]
    global b = s_aux[j+1]

    # If periodogram == false, sample on frequency at random,
    #                          otherwise get 'significant' frequencies.
    if (periodogram == false)
      n_freq_sample[j, 1] = 1
      ω_sample[j, 1:n_freq_sample[j, 1], 1] = rand(Uniform(0, 0.5), n_freq_sample[j, 1])
    else
      significant_ω = find_significant_ω(y[a:b], n_freq_max)
      n_freq_sample[j, 1] = length(significant_ω)
      ω_sample[j, 1:n_freq_sample[j, 1], 1] = significant_ω
    end

    σ_sample[j, 1] = 1

    y_aux = y[a:b]
    ω_aux = ω_sample[j, 1:n_freq_sample[j, 1], 1]
    X_aux = get_X(ω_aux, a, b)
    β_start_opt = inv(X_aux'*X_aux)*X_aux'*y_aux

    global ω = ω_aux
    global σ = σ_sample[j, 1]
    β_mean_sample[j, 1:(2*n_freq_sample[j, 1]+2), 1] =
         β_sample[j, 1:(2*n_freq_sample[j, 1]+2), 1] =
         optimize(neg_f_log_posterior_β, neg_g_log_posterior_β!,
          neg_h_log_posterior_β!, β_start_opt, BFGS()).minimizer


    log_likelik_sample[j, 1] = log_likelihood_segment(β_sample[j, 1:(2*n_freq_sample[j, 1]+2), 1],
                                                    ω_sample[j, 1:n_freq_sample[j, 1], 1],
                                                    a, b, σ_sample[j, 1])

  end


  output = Dict("β" => β_sample, "ω" => ω_sample, "σ" => σ_sample,
                "s" => s_sample, "n_freq" => n_freq_sample,
                "log_likelik" => log_likelik_sample,
                "β_mean" => β_mean_sample,
                "n_CP" => n_CP_sample)
end


















# --------------------------------------------------------------------------- #
# ----------------- Simulation Study, Illustrative Example ------------------ #
# --------------------------------------------------------------------------- #






# --------- Parameters Values, Section 5.1, Hadj-Amar et al. (2018) ------------


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



# ------------- Simulation Time Series  ------------

# Time Series
illustrative_example = get_data(n_obs, s_true, ω_true, β_true,
                                α_true, μ_true, σ_true)
data = illustrative_example["data"]
signal = illustrative_example["signal"]

# Plot: Observations and Signal
close(); scatter(1:n_obs, data, alpha = 0.7, s = 7)
plot(signal, color = "red", linestyle = "-")
[axvline(cp, color = "grey") for cp in s_true]



# ------------ Parameters MCMC for AutoNOM ----------


n_iter_MCMC = 10000 # number of MCMC iteration for AutoNOM
n_CP_max = 10 # maximum number of change-points
n_freq_max = 8 # maximum number of frequencies in each segment.

# -- Segment model:

σ_β = 10 # prior variance for β,  β ∼ Normal(0, σ_β * I)
ν0 = 1/100 # prior σ², InverseGamma(ν0/2, η0/2)
γ0 = 1/100 # prior σ², InverseGamma(ν0/2, η0/2)
λ_S = 1/10 # poisson prior parameter, i.e n of freq ∼ Poisson(λ_S)
δ_ω_mixing_brn = 0.2 # mixing probability random walk when relocation a frequency
#                        (after 300 iterations burn-in)
c_S = 0.4 # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
ϕ_ω = 0.25 # birth step, frequency is sampled from Unif(0, ψ_ω)
ψ_ω = 10/n_obs # miminum distance between frequencies
σ_RW_ω_hyper = 20 # variance parameter for random walk when relocating a freq :
#                   ωᵖ ∼ Normal(ωᶜ, σ_ω), where σ_ω = 1/(σ_RW_ω_hyper*n)


# -- Change-Point model:

λ_NS = 1\10 # poisson prior parameter, i.e n of cp ∼ Poisson(λ_NS)
c_NS = 0.4 # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
ψ_NS = 20 # minumum distance between change-points
σ_RW_s = 1 # variance parameter for random walk when relocating a change-point
δ_s_mixing = 0.2 # mixing probability for mixture proposal when relocating a change-point



# ----------- MCMC Objects and Starting Values -------------

global y = data
global n = length(data)

s_start = [100]
MCMC_objects = get_MCMC_objects(s_start)
# MCMC_objects = get_MCMC_objects(s_start, periodogram = true)

β_sample = MCMC_objects["β"]
ω_sample = MCMC_objects["ω"]
σ_sample = MCMC_objects["σ"]
s_sample = MCMC_objects["s"]
n_freq_sample = MCMC_objects["n_freq"]
log_likelik_sample = MCMC_objects["log_likelik"]
β_mean_sample = MCMC_objects["β_mean"]
n_CP_sample = MCMC_objects["n_CP"]




# ------------- AutoNOM - Automatic Nonstationary Oscillatory Modelling  --------


@showprogress for t in 2:(n_iter_MCMC + 1)

 global y = copy(data)
 global n = length(data)

 # Option: for the first 300 iterations,
 #         sample frequencies just from  periodogram (no Random Walk)

 if ( t <= 300 )
   global δ_ω_mixing = 1
 else
   global δ_ω_mixing = δ_ω_mixing_brn
 end

 n_CP_current = n_CP_sample[t-1]

 if (n_CP_current == 0)

   MCMC = birth_move_non_stationary(t, n_CP_sample, β_sample, β_mean_sample,
                ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample)
   n_CP_sample = MCMC["n_CP"]
   β_sample = MCMC["β"]
   β_mean_sample = MCMC["β_mean"]
   ω_sample = MCMC["ω"]
   σ_sample = MCMC["σ"]
   s_sample = MCMC["s"]
   n_freq_sample = MCMC["n_freq"]
   log_likelik_sample = MCMC["log_likelik"]

 elseif (n_CP_current ==  n_CP_max)

   death_prob_n_CP_max = c_NS*min(1, pdf(Poisson(λ_NS), n_CP_max-1)/pdf(Poisson(λ_NS), n_CP_max))
   within_prob_n_CP_max = 1 - death_prob_n_CP_max

   U = rand()

   if (U <= death_prob_n_CP_max)

     MCMC = death_move_non_stationary(t, n_CP_sample, β_sample, β_mean_sample,
                ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample)
     n_CP_sample = MCMC["n_CP"]
     β_sample = MCMC["β"]
     β_mean_sample = MCMC["β_mean"]
     ω_sample = MCMC["ω"]
     σ_sample = MCMC["σ"]
     s_sample = MCMC["s"]
     n_freq_sample = MCMC["n_freq"]
     log_likelik_sample = MCMC["log_likelik"]

   else

     MCMC = within_move_non_stationary(t, n_CP_sample, β_sample, β_mean_sample,
                ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample)
     n_CP_sample = MCMC["n_CP"]
     β_sample = MCMC["β"]
     β_mean_sample = MCMC["β_mean"]
     ω_sample = MCMC["ω"]
     σ_sample = MCMC["σ"]
     s_sample = MCMC["s"]
     n_freq_sample = MCMC["n_freq"]
     log_likelik_sample = MCMC["log_likelik"]

   end

 else

   # Probabilities for different types of moves
   birth_prob = c_NS*min(1, pdf(Poisson(λ_NS), n_CP_current+1)/pdf(Poisson(λ_NS), n_CP_current))
   death_prob = c_NS*min(1, pdf(Poisson(λ_NS), n_CP_current-1)/pdf(Poisson(λ_NS), n_CP_current))
   within_prob = 1 - (birth_prob + death_prob)

   U = rand()

   if (U <= birth_prob)

     MCMC = birth_move_non_stationary(t, n_CP_sample, β_sample, β_mean_sample,
                ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample)
     n_CP_sample = MCMC["n_CP"]
     β_sample = MCMC["β"]
     β_mean_sample = MCMC["β_mean"]
     ω_sample = MCMC["ω"]
     σ_sample = MCMC["σ"]
     s_sample = MCMC["s"]
     n_freq_sample = MCMC["n_freq"]
     log_likelik_sample = MCMC["log_likelik"]

   elseif ((U > birth_prob) && (U <= (birth_prob + death_prob)))

     MCMC = death_move_non_stationary(t, n_CP_sample, β_sample, β_mean_sample,
                ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample)
     n_CP_sample = MCMC["n_CP"]
     β_sample = MCMC["β"]
     β_mean_sample = MCMC["β_mean"]
     ω_sample = MCMC["ω"]
     σ_sample = MCMC["σ"]
     s_sample = MCMC["s"]
     n_freq_sample = MCMC["n_freq"]
     log_likelik_sample = MCMC["log_likelik"]


   else

     MCMC = within_move_non_stationary(t, n_CP_sample, β_sample, β_mean_sample,
                ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample)
     n_CP_sample = MCMC["n_CP"]
     β_sample = MCMC["β"]
     β_mean_sample = MCMC["β_mean"]
     ω_sample = MCMC["ω"]
     σ_sample = MCMC["σ"]
     s_sample = MCMC["s"]
     n_freq_sample = MCMC["n_freq"]
     log_likelik_sample = MCMC["log_likelik"]

   end

 end

 if (t % 1000 == 0)

   println("Iteration: ", t)

   println("Locations: ", s_sample[1:(n_CP_sample[t]), t])

   for j in 1:(n_CP_sample[t] + 1)

     println("Segment :", j)
     println("m: ", n_freq_sample[j, t])
     println("ω: ", ω_sample[j, 1:n_freq_sample[j, t], t])
     println("β: ", β_sample[j, 1:(2*n_freq_sample[j, t]+2), t])
     println("σ: ", σ_sample[j, t])

   end

 end
end




# ----------------------- Diagnostic Convergence  ----------------------


burn_in_MCMC = Int64(0.6*n_iter_MCMC)
final_indexes_MCMC = burn_in_MCMC:n_iter_MCMC

# Trace log likelihood
log_likelik_sample_total = zeros(Float64, n_iter_MCMC + 1)
for t in 1:(n_iter_MCMC+1)
    n_CP = n_CP_sample[t]
    log_likelik_sample_total[t] = sum(log_likelik_sample[1:(n_CP+1), t])
end

# -- Trace plot: log likelihood, for assessing convergence
close(); plot(log_likelik_sample_total[final_indexes_MCMC])

# -- Markov Chains after burn-in
n_CP_final = n_CP_sample[final_indexes_MCMC]
β_final = β_sample[:, :, final_indexes_MCMC]
ω_final = ω_sample[:, :, final_indexes_MCMC]
s_final = s_sample[:, final_indexes_MCMC]
σ_final = σ_sample[:, final_indexes_MCMC]
n_freq_final = n_freq_sample[:, final_indexes_MCMC]
n_final = length(n_CP_final)

# -- Trace plot: n of change-points
close(); plot(n_CP_final)

# Estimate: n of change-points
n_CP_est = mode(n_CP_final)
index_CP_est = find(n_CP_final .== n_CP_est)

# Estimate: n of frequencies, conditioned on n_CP_est
n_freq_est = zeros(Int64, n_CP_est+1)
for j in 1:(n_CP_est+1)
    n_freq_est[j] = mode(n_freq_final[j, index_CP_est])
end

# Histogram: n of frequencies for each segment
close();
for j in 1:(n_CP_est + 1)
  subplot(1, n_CP_est+1, j)
  title("Segment $j", fontsize = 10)
  plt[:hist](n_freq_final[j, index_CP_est], normed = true, color = "pink")
end


# Trace Plot: change-points location
close();
for j in 1:n_CP_est
    plot(s_final[j, index_CP_est], label = "s_$j")
end


# Estimate: change-points location
s_est = mean(s_final[1:n_CP_est, index_CP_est], 2)


# Trace Plot: frequencies in selected segment
# (conditioned on n change-points and n freq in each segment)
close()
segment = 1
for l in 1:n_freq_est[segment]
  index_CP_nfreq_est = find(n_freq_final[segment, :,  :][index_CP_est] .== n_freq_final[segment])
  subplot(n_freq_est[segment], 1, l)
  plot(ω_final[segment, l, index_CP_nfreq_est])
  axhline(ω_true[segment][l], color = "red", linestyle = "dotted")
end
suptitle("Segment $segment", fontsize = 15)



# Estimate: signal, by averaging over MCMC iterations
signal_MCMC = zeros(Float64, n_final, n_obs)
@showprogress for t in 1:n_final
  n_CP = n_CP_final[t]
  signal_MCMC[t, :] = get_estimated_signal(y, s_final[1:n_CP, t],
                                    β_final[:, :, t],
                                    ω_final[:, :, t],
                                    n_freq_final[:, t])
end
signal_est = reshape(mean(signal_MCMC, 1), n_obs)


# Plot: data, estimated signal, true signal and estimated change-point locations

close();
scatter(1:n_obs, data, alpha = 0.7, s = 7)
plot(signal_est, color = "red", linestyle = "-")
plot(signal, color = "green", linestyle = "dotted")
[axvline(cp, color = "grey") for cp in s_est]
