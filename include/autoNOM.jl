# ----- AutoNOM : Automatic Nonstationary Oscillatory Modelling : 

# ----INPUT :  

#     - data: time series

#     - hyperparms:
#                 hyperparms[:n_iter_MCMC] # number of MCMC iteration for AutoNOM
#                 hyperparms[:n_CP_max] # maximum number of change-points
#                 hyperparms[:n_freq_max] # maximum number of frequencies in each segment. 
#                 hyperparms[:σ_β] # prior variance for β,  β ∼ Normal(0, σ_β * I)
#                 hyperparms[:ν0] # prior σ², InverseGamma(ν0/2, η0/2)
#                 hyperparms[:γ0] # prior σ², InverseGamma(ν0/2, η0/2)
#                 hyperparms[:λ_S] # poisson prior parameter, i.e n of freq ∼ Poisson(λ_S)
#                 hyperparms[:δ_ω_mixing_brn] # mixing probability random walk when relocation a frequency
#                                             (after 300 iterations burn-in)
#                 hyperparms[:c_S] # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
#                 hyperparms[:ϕ_ω] # birth step, frequency is sampled from Unif(0, ψ_ω)
#                 hyperparms[:ψ_ω] # miminum distance between frequencies
#                 hyperparms[:σ_RW_ω_hyper]  # variance parameter for random walk when relocating a freq :
#                                            ωᵖ ∼ Normal(ωᶜ, σ_ω), where σ_ω = 1/(σ_RW_ω_hyper*n)
#                 hyperparms[:λ_NS] # poisson prior parameter, i.e n of cp ∼ Poisson(λ_NS)
#                 hyperparms[:c_NS]  # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
#                 hyperparms[:ψ_NS] = 10 # minumum distance between change-points
#                 hyperparms[:σ_RW_s] = 2.5 # variance parameter for random walk when relocating a change-point
#                 hyperparms[:δ_s_mixing] = 0.2 # mixing probability for mixture proposal when relocating a change-point
 
#     - s_start: starting value of change point location 


# -- OUTPUT :

#      - m: n freq sample 
#      - n_CP: n change point sample 
#      - s: change point location sample
#      - β: linear basis function coefficients sample 
#      - ω: frequencies sample
#      - log_likelik: log likelihood sample  



function AutoNOM(data, hyperparms; s_start = [40])

    global y = data
    global n = length(data)

    MCMC_objects = get_MCMC_objects(s_start, hyperparms, periodogram = true)

    β_sample = MCMC_objects["β"]
    ω_sample = MCMC_objects["ω"]
    σ_sample = MCMC_objects["σ"]
    s_sample = MCMC_objects["s"]
    n_freq_sample = MCMC_objects["n_freq"]
    log_likelik_sample = MCMC_objects["log_likelik"]
    n_CP_sample = MCMC_objects["n_CP"]

    @showprogress for t in 2:(hyperparms[:n_iter_MCMC] + 1)

        n_CP_max = hyperparms[:n_CP_max] # maximum number of change-points
        n_freq_max = hyperparms[:n_freq_max] # maximum number of frequencies in each segment. 
        δ_ω_mixing_brn = hyperparms[:δ_ω_mixing_brn] # mixing probability random walk when relocation a frequency
        λ_NS = hyperparms[:λ_NS] # poisson prior parameter, i.e n of cp ∼ Poisson(λ_NS)
        c_NS = hyperparms[:c_NS]  # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
       
       
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
       
          MCMC = birth_move_non_stationary(t, n_CP_sample, β_sample,
                       ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample, hyperparms)
          n_CP_sample = MCMC["n_CP"]
          β_sample = MCMC["β"]
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
       
            MCMC = death_move_non_stationary(t, n_CP_sample, β_sample,
                       ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample, hyperparms)
            n_CP_sample = MCMC["n_CP"]
            β_sample = MCMC["β"]
            ω_sample = MCMC["ω"]
            σ_sample = MCMC["σ"]
            s_sample = MCMC["s"]
            n_freq_sample = MCMC["n_freq"]
            log_likelik_sample = MCMC["log_likelik"]
       
          else
       
            MCMC = within_move_non_stationary(t, n_CP_sample, β_sample,
                       ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample, hyperparms)
            n_CP_sample = MCMC["n_CP"]
            β_sample = MCMC["β"]
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
       
            MCMC = birth_move_non_stationary(t, n_CP_sample, β_sample,
                       ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample, hyperparms)
            n_CP_sample = MCMC["n_CP"]
            β_sample = MCMC["β"]
            ω_sample = MCMC["ω"]
            σ_sample = MCMC["σ"]
            s_sample = MCMC["s"]
            n_freq_sample = MCMC["n_freq"]
            log_likelik_sample = MCMC["log_likelik"]
       
          elseif ((U > birth_prob) && (U <= (birth_prob + death_prob)))
       
            MCMC = death_move_non_stationary(t, n_CP_sample, β_sample,
                       ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample, hyperparms)
            n_CP_sample = MCMC["n_CP"]
            β_sample = MCMC["β"]
            ω_sample = MCMC["ω"]
            σ_sample = MCMC["σ"]
            s_sample = MCMC["s"]
            n_freq_sample = MCMC["n_freq"]
            log_likelik_sample = MCMC["log_likelik"]
       
       
          else
       
            MCMC = within_move_non_stationary(t, n_CP_sample, β_sample,
                       ω_sample, σ_sample, s_sample, n_freq_sample, log_likelik_sample, hyperparms)
            n_CP_sample = MCMC["n_CP"]
            β_sample = MCMC["β"]
            ω_sample = MCMC["ω"]
            σ_sample = MCMC["σ"]
            s_sample = MCMC["s"]
            n_freq_sample = MCMC["n_freq"]
            log_likelik_sample = MCMC["log_likelik"]
       
          end
       
        end
       
        if (t % 2000 == 0)
       
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
    
       return Dict(:m => n_freq_sample, :ω => ω_sample, :β => β_sample, :σ => σ_sample, 
                    :n_CP => n_CP_sample,
                   :s => s_sample, :log_likelik => log_likelik_sample)
    
end 