function get_data(n_obs, parms_true)

    ω = parms_true[:ω]
    β = parms_true[:β] 
    s = parms_true[:s] 
    α = parms_true[:α]
    μ = parms_true[:μ]
    σ = parms_true[:σ]

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

    return Dict(:data => signal + noise, :signal => signal)
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

function get_MCMC_objects(s, hyperparms; periodogram = false)

  n_iter_MCMC = hyperparms[:n_iter_MCMC]
  n_CP_max = hyperparms[:n_CP_max]
  n_freq_max = hyperparms[:n_freq_max] 
  σ_β = hyperparms[:σ_β]
  n_obs = length(y);

  # --- MCMC Objects

  β_sample = zeros(Float64, n_CP_max+1, 2*n_freq_max+2, n_iter_MCMC + 1)
  ω_sample = zeros(Float64, n_CP_max+1, n_freq_max, n_iter_MCMC + 1)
  s_sample = zeros(Float64, n_CP_max, n_iter_MCMC + 1)
  σ_sample = zeros(Float64, n_CP_max + 1, n_iter_MCMC + 1)
  n_freq_sample = zeros(Int64, n_CP_max+1, n_iter_MCMC+1)
  log_likelik_sample = zeros(Float64, n_CP_max + 1, n_iter_MCMC + 1)
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
    β_var_aux = inv(eye(2*n_freq_sample[j, 1]+2)/(σ_β^2) + (X_aux'*X_aux)/(σ_sample[j, 1]^2))
    β_var_aux = 0.5*(β_var_aux + β_var_aux')
    β_mean_aux = β_var_aux*((X_aux'*y_aux)/(σ_sample[j, 1]^2))
    β_sample[j, 1:(2*n_freq_sample[j, 1]+2), 1] = rand(MultivariateNormal(β_mean_aux, β_var_aux), 1)

    log_likelik_sample[j, 1] = log_likelihood_segment(β_sample[j, 1:(2*n_freq_sample[j, 1]+2), 1],
                                                    ω_sample[j, 1:n_freq_sample[j, 1], 1],
                                                    a, b, σ_sample[j, 1])

  end


  output = Dict("β" => β_sample, "ω" => ω_sample, "σ" => σ_sample,
                "s" => s_sample, "n_freq" => n_freq_sample,
                "log_likelik" => log_likelik_sample,
                "n_CP" => n_CP_sample)
end



# Function: for starting values. get_significant ω, by testing w.r.t to true distribution (Fisher test)
# using smoothed periodogram.
function find_significant_ω(y, n_freq_max)

    ω = []
    n = length(y)
    y_demean = y .- mean(y)
  
    n = length(y_demean)
    period = DSP.periodogram(y_demean)
  
    I = period.power[2:(end-1)]
    freq = period.freq[2:(end-1)]
  
    freq_test = freq[[x[2] for x in findpeaks(I, n_freq_max)]]
    I_test = [x[1] for x in findpeaks(I, n_freq_max)]
    a = floor(Int64, (n-1)/2)
  
    for i in 1:length(I_test)
  
      g_test = I_test[i]/sum(I)
      b = floor(Int64, 1/g_test)
      b_max = get_max_b(a, b)
      x = copy(g_test)
      p_val = get_p_value_freq(x, a, b_max)
  
      if (p_val <= 5e-2 && p_val >= 0)
        push!(ω, freq_test[i])
      end
  
    end
  
    return ω
end



# Function: for starting values. findpeaks of a 1D array, sorted in decreasing order (of power)
function findpeaks(A, n_peaks)

    maximums = Tuple{Float64, Int}[]
    n = length(A)
  
    if (A[1] > A[2])
      push!(maximums, (A[1], 1))
    end
  
    if (A[n] > A[n-1])
      push!(maximums, (A[n], n))
    end
  
    for j in 2:(n-1)
      if (A[j-1] < A[j] && A[j+1] < A[j])
        push!(maximums, (A[j], j))
      end
    end
  
    maximums = sort(maximums, by = x->x[1], rev = true)[1:min(n_peaks, length(maximums))]
  
    return maximums
  end


# Function: for starting values. get maximum value of b for evaluating Binomial(a, i),
#           for i = 1, ...b.
function get_max_b(a, b)
    b_max = 1
    try
      for i in 1:b
        binomial(a, i)
        b_max += 1
      end
      return (b_max - 1)
    catch
      return (b_max - 1)
    end
end


# Function: for starting values. get p_value for Fisher test, for the largest periodogram ordinate
# if it is not possible to evaluate binom(a, b) it will return (-1).
function get_p_value_freq(x, a::Int64, b::Int64)
    out = 0.0
    for i in 1:b
      out += ((-1)^(i-1))*binomial(a, i)*((1-i*x)^(a-1))
    end
    return out
end
  

# - 
function eye(n) 
    return 1.0*Matrix(I,n,n)
end


# Function: log_posterior_β
function log_posterior_β(β, ω, a, b, σ, σ_β)

  X = get_X(ω, a, b)
  y_sub = y[a:b]
  f = ((-sum((y_sub - X*β).^2)/(2*(σ^2))) - ((β'*β)/(2*(σ_β^2))))[1]

  return f
end


# Function: log_likelihood within segement
function log_likelihood_segment(β, ω, a, b, σ)

  X_sub = get_X(ω, a, b)
  y_sub = y[a:b]
  n_sub = length(y_sub)

  out = (-n_sub/2)*log((2π*(σ^2))) -
        (sum((y_sub - X_sub*β).^2))/(2*(σ^2))

  return out
end

# Function: log_likelihood within segment auxiliary
function log_likelihood_segment_aux(y_sub, β, ω, a, b, σ)

  X_sub = get_X(ω, a, b)
  n_sub = length(y_sub)

  out = (-n_sub/2)*log((2π*(σ^2))) -
        (sum((y_sub - X_sub*β).^2))/(2*(σ^2))

  return out
end

# Function: check whether is possible to perform a birth move.
#           i.e. it checks whether the support of s, constrained by ψ_NS is > 0.
function is_birth_possible(s_current, n_CP_current)

    support_s = Array(Vector{Float64}, n_CP_current + 1)
    for k in 1:(n_CP_current + 1)
      support_s[k] = [s_current[k]+ψ_NS, s_current[k+1]-ψ_NS]
    end
    length_support = get_support_s(support_s)

    if (length_support == 0.0) return false end
    return true
end

# Function: get the support of s given the constraint imposed by ϕ.
function get_support_s(support_s)
  out = 0.0
  for i in 1:length(support_s)
    if ((support_s[i][2] - support_s[i][1])>=0)
      aux = support_s[i][2] - support_s[i][1]
    else
      aux = 0.0
    end
    out += aux
  end
  return out
end


# Function: get signal estimate
function get_estimated_signal(y, s, β, ω, n_freq)

  n = length(y)
  s_floor = floor.(Int64, s)
  s_aux = vcat(1, s_floor, n)

  y_fit = []

  for k in 1:(length(s) + 1)
    a = s_aux[k]
    if (k == (length(s) + 1))
      y_sub = y[s_aux[k]:(s_aux[k+1])]
      b = s_aux[k+1]
    else
      y_sub = y[s_aux[k]:(s_aux[k+1]-1)]
      b = s_aux[k+1]-1
    end

    n_sub = length(y_sub)
    ω_sub = ω[k, 1:n_freq[k]]
    X_sub = get_X(ω_sub, a, b)
    β_sub = β[k, 1:(2*n_freq[k]+2)]
    y_fit_sub = X_sub*β_sub
    y_fit = vcat(y_fit, y_fit_sub)
  end

  return y_fit
end


# Functon: non-stationary model, within step.
function within_move_non_stationary(t, n_CP_sample, β_sample, ω_sample,
                          σ_sample, s_sample, n_freq_sample, log_likelik_sample, 
                          hyperparms)

  λ_NS = hyperparms[:λ_NS] # poisson prior parameter, i.e n of cp ∼ Poisson(λ_NS)
  c_NS = hyperparms[:c_NS]  # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
  ψ_NS =hyperparms[:ψ_NS] # minumum distance between change-points 
  σ_β = hyperparms[:σ_β]  
  σ_RW_s = hyperparms[:σ_RW_s]   
  δ_s_mixing = hyperparms[:δ_s_mixing] 
  n_CP_max = hyperparms[:n_CP_max] # maximum number of change-points
  n_freq_max = hyperparms[:n_freq_max] # maximum number of frequencies in each segment. 
  ν0 = hyperparms[:ν0] # prior σ², InverseGamma(ν0/2, η0/2)
  γ0 = hyperparms[:γ0] # prior σ², InverseGamma(ν0/2, η0/2)     
  λ_S = hyperparms[:λ_S] # poisson prior parameter, i.e n of freq ∼ Poisson(λ_S)
  c_S = hyperparms[:c_S] # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
  ϕ_ω = hyperparms[:ϕ_ω] # birth step, frequency is sampled from Unif(0, ψ_ω)
  ψ_ω = hyperparms[:ψ_ω] # miminum distance between frequencies
  σ_RW_ω_hyper = hyperparms[:σ_RW_ω_hyper]  # variance parameter for random walk when relocating a freq :
            
            

  global y = data
  global n = length(data)

  n_CP_current = n_CP_sample[t-1]
  β_seg = β_sample[:, :, t-1]
  # β_mean_seg = β_mean_sample[:, :, t-1]
  ω_seg = ω_sample[:, :, t-1]
  σ_seg = σ_sample[1:(n_CP_current + 1), t-1]
  s = s_sample[1:n_CP_current, t-1]
  n_freq_seg = n_freq_sample[1:(n_CP_current +1), t-1]
  log_likelik_seg = log_likelik_sample[1:(n_CP_current +1), t-1]

  s_current = vcat(1, s, n)

  U = rand()

  if ( U <= δ_s_mixing)

    aux = -1
    while (aux < 0 )
        global index_s = sample(1:n_CP_current)
        aux = (s_current[index_s+2] - ψ_NS - (s_current[index_s] + ψ_NS))
    end

    s_select = s_current[index_s+1]

    n1 = floor(Int64, s_current[index_s+1]) - floor(Int64, s_current[index_s])
    n2 = floor(Int64, s_current[index_s+2]) - floor(Int64, s_current[index_s+1])

    a_1 = floor(Int64, s_current[index_s])
    b_1 = floor(Int64, s_current[index_s+1]-1)
    a_2 = floor(Int64, s_current[index_s+1])
    if (s_current[index_s+2] == n)
      b_2 = floor(Int64, s_current[index_s+2])
    else
      b_2 = floor(Int64, s_current[index_s+2]) - 1
    end

    y_curr_1 = y[a_1:b_1]
    y_curr_2 = y[a_2:b_2]

    s_star = rand(Uniform(s_current[index_s] + ψ_NS, s_current[index_s+2] - ψ_NS))

  else

    global index_s = sample(1:n_CP_current)
    s_select = s_current[index_s+1]

    n1 = floor(Int64, s_current[index_s+1]) - floor(Int64, s_current[index_s])
    n2 = floor(Int64, s_current[index_s+2]) - floor(Int64, s_current[index_s+1])

    a_1 = floor(Int64, s_current[index_s])
    b_1 = floor(Int64, s_current[index_s+1]-1)
    a_2 = floor(Int64, s_current[index_s+1])
    if (s_current[index_s+2] == n)
      b_2 = floor(Int64, s_current[index_s+2])
    else
      b_2 = floor(Int64, s_current[index_s+2]) - 1
    end

    y_curr_1 = y[a_1:b_1]
    y_curr_2 = y[a_2:b_2]

    if ((n1 > ψ_NS) && (n2 > ψ_NS))
      s_star = rand(Normal(s_select, σ_RW_s), 1)[1]
    elseif ((n1 == ψ_NS) && (n2 > ψ_NS))
      s_star = rand(Normal(s_select, σ_RW_s), 1)[1]
    elseif ((n1 > ψ_NS) && (n2 == ψ_NS))
      s_star = rand(Normal(s_select, σ_RW_s), 1)[1]
    else
      s_star = s_select
    end
  end


  s_proposed = float(copy(s_current))
  s_proposed[index_s+1] = s_star

  ω_current_1 = ω_seg[index_s, 1:n_freq_seg[index_s]]
  ω_current_2 = ω_seg[index_s+1, 1:n_freq_seg[index_s+1]]
  β_current_1 = β_seg[index_s, 1:(2*n_freq_seg[index_s]+2)]
  β_current_2 = β_seg[index_s+1, 1:(2*n_freq_seg[index_s+1]+2)]

  σ_current_1 = σ_seg[index_s]
  σ_current_2 = σ_seg[index_s+1]

  n_freq_curr_1 = length(ω_current_1)
  n_freq_curr_2 = length(ω_current_2)

  log_likelik_1_current = log_likelik_seg[index_s]
  log_likelik_2_current = log_likelik_seg[index_s + 1]

  # Proposed omega are the same as the previous partitions
  ω_star_1 = copy(ω_current_1)
  n_freq_prop_1 = length(ω_star_1)
  ω_star_2 = copy(ω_current_2)
  n_freq_prop_2 = length(ω_star_2)

  # -- Proposing β, partition 1.

  a = floor(Int64, s_proposed[index_s])
  b = floor(Int64, s_proposed[index_s + 1]) - 1
  y_star_1 = y[a:b]

  X_prop_1 = get_X(ω_star_1, a, b)
  β_var_prop_1 = inv(eye(2*n_freq_prop_1 + 2)/(σ_β^2) + (X_prop_1'*X_prop_1)/(σ_current_1^2))
  β_var_prop_1 = 0.5*(β_var_prop_1 + β_var_prop_1')
  β_mean_prop_1 = β_var_prop_1*((X_prop_1'*y_star_1)/(σ_current_1^2))
  β_proposed_1 = rand(MvNormal(β_mean_prop_1, β_var_prop_1), 1)

  log_likelik_1_proposed = log_likelihood_segment(β_proposed_1, ω_star_1,
                                                  a, b, σ_current_1)


  # -- Proposing β, partition 2

  a = floor(Int64, s_proposed[index_s + 1])
  if (s_proposed[index_s + 2] == n)
    b = floor(Int64, s_proposed[index_s + 2])
  else
    b = floor(Int64, s_proposed[index_s + 2]) - 1
  end
  y_star_2 = y[a:b]


  X_prop_2 = get_X(ω_star_2, a, b)
  β_var_prop_2 = inv(eye(2*n_freq_prop_2 + 2)/(σ_β^2) + (X_prop_2'*X_prop_2)/(σ_current_2^2))
  β_var_prop_2 = 0.5*(β_var_prop_2 + β_var_prop_2')
  β_mean_prop_2 = β_var_prop_2*((X_prop_2'*y_star_2)/(σ_current_2^2))
  β_proposed_2 = rand(MvNormal(β_mean_prop_2, β_var_prop_2), 1)

  log_likelik_2_proposed = log_likelihood_segment(β_proposed_2, ω_star_2, a, b, σ_current_2)





  # ---- Auxiliary for proposal ratio β

  X_curr_1 = get_X(ω_current_1, a_1, b_1)
  β_var_curr_1 = inv(eye(2*n_freq_curr_1 + 2)/(σ_β^2) + (X_curr_1'*X_curr_1)/(σ_current_1^2))
  β_var_curr_1 = 0.5*(β_var_curr_1 + β_var_curr_1')
  β_mean_curr_1 = β_var_curr_1*((X_curr_1'*y_curr_1)/(σ_current_1^2))

  X_curr_2 = get_X(ω_current_2, a_2, b_2)
  β_var_curr_2 = inv(eye(2*n_freq_curr_2 + 2)/(σ_β^2) + (X_curr_2'*X_curr_2)/(σ_current_2^2))
  β_var_curr_2 = 0.5*(β_var_curr_2 + β_var_curr_2')
  β_mean_curr_2 = β_var_curr_2*((X_curr_2'*y_curr_2)/(σ_current_2^2))



  # ------------ Log_likelik_ratio

  log_likelik_proposed = log_likelik_1_proposed + log_likelik_2_proposed
  log_likelik_current = log_likelik_1_current + log_likelik_2_current
  log_likelik_ratio = log_likelik_proposed - log_likelik_current


  # ------------ Prior_ratio
  # - s
  log_prior_s_proposed = log(s_proposed[index_s + 2] - s_proposed[index_s + 1])  +
                         log(s_proposed[index_s + 1] - s_proposed[index_s])
  log_prior_s_current = log(s_current[index_s + 2] - s_current[index_s + 1])  +
                         log(s_current[index_s + 1] - s_current[index_s])
  log_prior_s_ratio = log_prior_s_proposed - log_prior_s_current

  log_prior_ratio = copy(log_prior_s_ratio)




  # ----------- Proposal_ratio


  log_proposal_β_1_curr = (log.(pdf(MvNormal(β_mean_curr_1, β_var_curr_1),
                            β_current_1))[1])
  log_proposal_β_2_curr = (log.(pdf(MvNormal(β_mean_curr_2, β_var_curr_2),
                            β_current_2))[1])
  log_proposal_β_curr = log_proposal_β_1_curr + log_proposal_β_2_curr


  log_proposal_β_1_prop = (log.(pdf(MvNormal(β_mean_prop_1, β_var_prop_1),
                            β_proposed_1))[1])
  log_proposal_β_2_prop = (log.(pdf(MvNormal(β_mean_prop_2, β_var_prop_2),
                            β_proposed_2))[1])
  log_proposal_β_prop = log_proposal_β_1_prop + log_proposal_β_2_prop


  log_proposal_ratio = log_proposal_β_curr - log_proposal_β_prop



  # ---------- Output proposed objects (except σ)

  if (index_s == 1)

    # β
    β_aux_1 = β_seg[(index_s+2):(n_CP_current + 1), :]
    β_aux_2 = reshape(vcat(vec(β_proposed_1), zeros((2*n_freq_max+2) - length(β_proposed_1))), 1, (2*n_freq_max+2))
    β_aux_3 = reshape(vcat(vec(β_proposed_2), zeros((2*n_freq_max+2) - length(β_proposed_2))), 1, (2*n_freq_max+2))
    β_aux = [β_aux_2; β_aux_3; β_aux_1]
    β_zeros_aux = zeros((n_CP_max - size(β_aux)[1] + 1), (2*n_freq_max+2))
    β_proposed = [β_aux; β_zeros_aux]

    # ω
    ω_aux_1 = ω_seg[(index_s+2):(n_CP_current + 1), :]
    ω_aux_2 = reshape(vcat(vec(ω_star_1), zeros(n_freq_max - length(ω_star_1))), 1, n_freq_max)
    ω_aux_3 = reshape(vcat(vec(ω_star_2), zeros(n_freq_max - length(ω_star_2))), 1, n_freq_max)
    ω_aux =  [ω_aux_2; ω_aux_3; ω_aux_1;]
    ω_zeros_aux = zeros((n_CP_max - size(ω_aux)[1] + 1), n_freq_max)
    ω_proposed =[ω_aux; ω_zeros_aux]

    # n_freq
    n_freq_aux_1 = n_freq_seg[(index_s+2):(n_CP_current + 1)]
    n_freq_aux_2 = vcat(n_freq_prop_1, n_freq_prop_2)
    n_freq_aux = [n_freq_aux_2; n_freq_aux_1]
    n_freq_zeros_aux = zeros(Int64, (n_CP_max - size(n_freq_aux)[1] + 1))
    n_freq_proposed = [n_freq_aux; n_freq_zeros_aux]

    # log_likelik
    log_likelik_aux_1  = log_likelik_seg[(index_s+2):(n_CP_current + 1)]
    log_likelik_aux_2 = vcat(log_likelik_1_proposed, log_likelik_2_proposed )
    log_likelik_aux = [log_likelik_aux_2; log_likelik_aux_1;]
    log_likelik_zeros_aux = zeros(Int64, (n_CP_max - size(log_likelik_aux)[1] + 1))
    log_likelik_seg_proposed = [log_likelik_aux; log_likelik_zeros_aux]

  elseif (index_s == n_CP_current)

    # β
    β_aux_1 = β_seg[1:(index_s-1), :]
    β_aux_2 = reshape(vcat(vec(β_proposed_1), zeros((2*n_freq_max+2) - length(β_proposed_1))), 1, (2*n_freq_max+2))
    β_aux_3 = reshape(vcat(vec(β_proposed_2), zeros((2*n_freq_max+2) - length(β_proposed_2))), 1, (2*n_freq_max+2))
    β_aux = [β_aux_1; β_aux_2; β_aux_3]
    β_zeros_aux = zeros((n_CP_max - size(β_aux)[1] + 1), (2*n_freq_max+2))
    β_proposed = [β_aux; β_zeros_aux]


    # ω
    ω_aux_1 = ω_seg[1:(index_s-1), :]
    ω_aux_2 = reshape(vcat(vec(ω_star_1), zeros(n_freq_max - length(ω_star_1))), 1, n_freq_max)
    ω_aux_3 = reshape(vcat(vec(ω_star_2), zeros(n_freq_max - length(ω_star_2))), 1, n_freq_max)
    ω_aux = [ω_aux_1; ω_aux_2; ω_aux_3;]
    ω_zeros_aux = zeros((n_CP_max - size(ω_aux)[1] + 1), n_freq_max)
    ω_proposed =[ω_aux; ω_zeros_aux]

    # n_freq
    n_freq_aux_1 = n_freq_seg[1:(index_s-1)]
    n_freq_aux_2 = vcat(n_freq_prop_1, n_freq_prop_2)
    n_freq_aux = [n_freq_aux_1; n_freq_aux_2]
    n_freq_zeros_aux = zeros(Int64, (n_CP_max - size(n_freq_aux)[1] + 1))
    n_freq_proposed = [n_freq_aux; n_freq_zeros_aux]

    # log_likelik
    log_likelik_aux_1  = log_likelik_seg[1:(index_s-1)]
    log_likelik_aux_2 = vcat(log_likelik_1_proposed, log_likelik_2_proposed )
    log_likelik_aux = [log_likelik_aux_1; log_likelik_aux_2]
    log_likelik_zeros_aux = zeros(Int64, (n_CP_max - size(log_likelik_aux)[1] + 1))
    log_likelik_seg_proposed = [log_likelik_aux; log_likelik_zeros_aux]
  else

    # β
    β_aux_1 = β_seg[1:(index_s-1), :]
    β_aux_2 = reshape(vcat(vec(β_proposed_1), zeros((2*n_freq_max+2) - length(β_proposed_1))), 1, (2*n_freq_max+2))
    β_aux_3 = reshape(vcat(vec(β_proposed_2), zeros((2*n_freq_max+2) - length(β_proposed_2))), 1, (2*n_freq_max+2))
    β_aux_4 = β_seg[(index_s+2:(n_CP_current+1)), :]
    β_aux = [β_aux_1; β_aux_2; β_aux_3; β_aux_4;]
    β_zeros_aux = zeros((n_CP_max - size(β_aux)[1] + 1), (2*n_freq_max+2))
    β_proposed = [β_aux; β_zeros_aux]



    # ω
    ω_aux_1 = ω_seg[1:(index_s-1), :]
    ω_aux_2 = reshape(vcat(vec(ω_star_1), zeros(n_freq_max - length(ω_star_1))), 1, n_freq_max)
    ω_aux_3 = reshape(vcat(vec(ω_star_2), zeros(n_freq_max - length(ω_star_2))), 1, n_freq_max)
    ω_aux_4 = ω_seg[(index_s+2:(n_CP_current+1)), :]
    ω_aux = [ω_aux_1; ω_aux_2; ω_aux_3; ω_aux_4; ]
    ω_zeros_aux = zeros((n_CP_max - size(ω_aux)[1] + 1), n_freq_max)
    ω_proposed =[ω_aux; ω_zeros_aux]

    # n_freq
    n_freq_aux_1 = n_freq_seg[1:(index_s-1)]
    n_freq_aux_2 = vcat(n_freq_prop_1, n_freq_prop_2)
    n_freq_aux_3 = n_freq_seg[(index_s+2:(n_CP_current+1))]
    n_freq_aux = [n_freq_aux_1; n_freq_aux_2; n_freq_aux_3]
    n_freq_zeros_aux = zeros(Int64, (n_CP_max - size(n_freq_aux)[1] + 1))
    n_freq_proposed = [n_freq_aux; n_freq_zeros_aux]

    # log_likelik
    log_likelik_aux_1  = log_likelik_seg[1:(index_s-1)]
    log_likelik_aux_2 = vcat(log_likelik_1_proposed, log_likelik_2_proposed )
    log_likelik_aux_3 = log_likelik_seg[(index_s+2:(n_CP_current+1))]
    log_likelik_aux = [log_likelik_aux_1; log_likelik_aux_2; log_likelik_aux_3]
    log_likelik_zeros_aux = zeros(Int64, (n_CP_max - size(log_likelik_aux)[1] + 1))
    log_likelik_seg_proposed = [log_likelik_aux; log_likelik_zeros_aux]
  end

  # --- Accept / Reject
  MH_ratio = log_likelik_ratio + log_prior_ratio + log_proposal_ratio
  epsilon = min(1, exp(MH_ratio))
  U = rand()


  # ---- # Accept / Reject

  if (U <= epsilon)
    n_CP_sample[t] = n_CP_current
    s_sample[1:n_CP_sample[t], t] = s_proposed[2:(end-1)]
    β_sample[:, :, t] = β_proposed
    # β_mean_sample[:, :, t] = β_mean_proposed
    ω_sample[:, :, t] = ω_proposed
    n_freq_sample[:, t] = n_freq_proposed
    log_likelik_sample[:, t] = log_likelik_seg_proposed
  else
    n_CP_sample[t] = n_CP_current
    s_sample[1:n_CP_sample[t], t] = s_current[2:(end-1)]
    β_sample[:, :, t] = β_seg
    # β_mean_sample[:, :, t] = β_mean_seg
    ω_sample[:, :, t] = ω_seg
    n_freq_sample[1:(n_CP_sample[t] + 1), t] = n_freq_seg
    log_likelik_sample[1:(n_CP_sample[t] + 1), t] = log_likelik_seg
  end



  # ---- Updating σ
  β_seg_aux_current = β_sample[:, :, t]
  ω_seg_aux_current = ω_sample[:, :, t]
  n_freq_seg_aux_current = n_freq_sample[1:(n_CP_current +1), t]
  s_aux_current = vcat(1, s_sample[1:n_CP_sample[t], t], n)

  n1_aux_current = floor(Int64, s_aux_current[index_s+1]) - floor(Int64, s_aux_current[index_s])
  n2_aux_current = floor(Int64, s_aux_current[index_s+2]) - floor(Int64, s_aux_current[index_s+1])
  ω1_aux_current = ω_seg_aux_current[index_s, 1:n_freq_seg_aux_current[index_s]]
  ω2_aux_current = ω_seg_aux_current[index_s+1, 1:n_freq_seg_aux_current[index_s+1]]
  β1_aux_current = β_seg_aux_current[index_s, 1:(2*n_freq_seg_aux_current[index_s]+2)]
  β2_aux_current = β_seg_aux_current[index_s+1, 1:(2*n_freq_seg_aux_current[index_s+1]+2)]


  a = floor(Int64, s_aux_current[index_s])
  b = floor(Int64, s_aux_current[index_s + 1]) - 1
  y_aux_current_1 = y[a:b]
  X_aux = get_X(ω1_aux_current, a, b)
  res_var_1 = sum((y_aux_current_1 - X_aux*β1_aux_current).^2)


  a = floor(Int64, s_aux_current[index_s + 1])
  if (s_aux_current[index_s + 2] == n)
    b = floor(Int64, s_aux_current[index_s + 2])
  else
    b = floor(Int64, s_aux_current[index_s + 2]) - 1
  end

  y_aux_current_2 = y[a:b]
  X_aux = get_X(ω2_aux_current, a, b)
  res_var_2 = sum((y_aux_current_2 - X_aux*β2_aux_current).^2)

  ν_post_1 = (n1_aux_current + ν0)/2
  ν_post_2 = (n2_aux_current + ν0)/2

  γ_post_1 = (γ0 + res_var_1)/2
  γ_post_2  = (γ0 + res_var_2)/2

  σ_proposed_1 = sqrt(rand(InverseGamma(ν_post_1, γ_post_1), 1)[1])
  σ_proposed_2 = sqrt(rand(InverseGamma(ν_post_2, γ_post_2), 1)[1])


  if (index_s == 1)

    # σ
    σ_aux_1 = σ_seg[(index_s+2):(n_CP_current + 1)]
    σ_aux_2 = vcat(σ_proposed_1, σ_proposed_2)
    σ_aux = [σ_aux_2; σ_aux_1]
    σ_zeros_aux = zeros(Float64, (n_CP_max - size(σ_aux)[1] + 1))
    σ_sample[:, t] = [σ_aux; σ_zeros_aux]

  elseif (index_s == n_CP_current)

      # σ
    σ_aux_1 = σ_seg[1:(index_s-1)]
    σ_aux_2 = vcat(σ_proposed_1, σ_proposed_2)
    σ_aux = [σ_aux_1; σ_aux_2]
    σ_zeros_aux = zeros(Float64, (n_CP_max - size(σ_aux)[1] + 1))
    σ_sample[:, t] = [σ_aux; σ_zeros_aux]

  else

    # n_freq
    σ_aux_1 = σ_seg[1:(index_s-1)]
    σ_aux_2 = vcat(σ_proposed_1, σ_proposed_2)
    σ_aux_3 = σ_seg[(index_s+2:(n_CP_current+1))]
    σ_aux = [σ_aux_1; σ_aux_2; σ_aux_3]
    σ_zeros_aux = zeros(Float64, (n_CP_max - size(σ_aux)[1] + 1))
    σ_sample[:, t]  = [σ_aux; σ_zeros_aux]

  end

  # Updating parameters (Segment model) for each regime
  s_aux = vcat(1, s_sample[1:n_CP_current, t], n_obs)

  for j in 1:(n_CP_current + 1)
    a = floor(Int64, s_aux[j])
    if (j == n_CP_current + 1)
      b = floor(Int64, n_obs)
    else
      b = floor(Int64, (s_aux[j+1] - 1))
    end
    global y = data[a:b]
    global n = length(y)
    global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

    m_current = n_freq_sample[j, t]
    β_current = β_sample[j, 1:(2*m_current+2), t]
    ω_current = ω_sample[j, 1:(m_current), t]
    σ_current = σ_sample[j, t]

    MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
    n_freq_sample[j, t] = MCMC_S["m"]
    β_sample[j, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2) - length(MCMC_S["β"])))
    ω_sample[j, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
    σ_sample[j, t] = MCMC_S["σ"]
    log_likelik_sample[j, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

  end


  output = Dict("n_CP" => n_CP_sample, "β" => β_sample,  "σ" => σ_sample,
                "ω" => ω_sample, "s" => s_sample, "n_freq" => n_freq_sample, "log_likelik" => log_likelik_sample)

  return output
end

# Function: non-stationary model, birth step.
function birth_move_non_stationary(t, n_CP_sample, β_sample, ω_sample,
                        σ_sample, s_sample, n_freq_sample, log_likelik_sample, 
                        hyperparms)


  λ_NS = hyperparms[:λ_NS] # poisson prior parameter, i.e n of cp ∼ Poisson(λ_NS)
  c_NS = hyperparms[:c_NS]  # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
  ψ_NS =hyperparms[:ψ_NS] # minumum distance between change-points 
  σ_β = hyperparms[:σ_β]  
  σ_RW_s = hyperparms[:σ_RW_s]   
  δ_s_mixing = hyperparms[:δ_s_mixing] 
  n_CP_max = hyperparms[:n_CP_max] # maximum number of change-points
  n_freq_max = hyperparms[:n_freq_max] # maximum number of frequencies in each segment. 
  ν0 = hyperparms[:ν0] # prior σ², InverseGamma(ν0/2, η0/2)
  γ0 = hyperparms[:γ0] # prior σ², InverseGamma(ν0/2, η0/2)     
  λ_S = hyperparms[:λ_S] # poisson prior parameter, i.e n of freq ∼ Poisson(λ_S)
  c_S = hyperparms[:c_S] # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
  ϕ_ω = hyperparms[:ϕ_ω] # birth step, frequency is sampled from Unif(0, ψ_ω)
  ψ_ω = hyperparms[:ψ_ω] # miminum distance between frequencies
  σ_RW_ω_hyper = hyperparms[:σ_RW_ω_hyper]  # variance parameter for random walk when relocating a freq :
              
         

  global y = data
  global n = length(data)

  n_CP_current = n_CP_sample[t-1]
  β_seg = β_sample[:, :, t-1]
  ω_seg = ω_sample[:, :, t-1]
  σ_seg = σ_sample[1:n_CP_current+1, t-1]
  s = s_sample[1:n_CP_current, t-1]
  n_freq_seg = n_freq_sample[1:(n_CP_current +1), t-1]
  log_likelik_seg = log_likelik_sample[1:(n_CP_current +1), t-1]

  if (s == 0.0)
    s_current = vcat(1, n)
  else
    s_current = vcat(1, s, n)
  end

  # Support of s, constrained by ψ
  support_s = Array{Vector{Float64}}(undef, n_CP_current + 1)
  for k in 1:(n_CP_current + 1)
    support_s[k] = [s_current[k]+ψ_NS, s_current[k+1]-ψ_NS]
  end
  length_support = get_support_s(support_s)

  # --- Sampling s*, proposed changing point (birth)
  s_star = sample_uniform_continuous_intervals(1, support_s)[1]
  s_proposed = sort(vcat(s_current, s_star))
  index_s = (findall(s_proposed .== s_star) .- 1)[1]


  #  --------- Current parameters needed for evaluating acceptance probability ----------
  ω_current = ω_seg[index_s, 1:n_freq_seg[index_s]]
  n_freq_current = length(ω_current)
  β_current = β_seg[index_s, 1:(2*n_freq_seg[index_s]+2)]
  σ_current = σ_seg[index_s]
  log_likelik_current = log_likelik_seg[index_s]

  ######### ------ Proposing σ's ---- ######'

  u = rand()
  σ_proposed_1 = sqrt((σ_current^2) * (u/(1-u)))
  σ_proposed_2 = sqrt((σ_current^2) * ((1-u)/u))



  # --- Proposing beta for new partition 1

  a = floor(Int64, s_proposed[index_s])
  b = floor(Int64, s_proposed[index_s + 1]) - 1
  y_star_1 = y[a:b]
  n_obs_prop_1  = length(y_star_1)
  ω_star_1 = copy(ω_current)
  n_freq_prop_1 = length(ω_star_1)

  X_prop_1 = get_X(ω_star_1, a, b)
  β_var_prop_1 = inv(eye(2*n_freq_prop_1 + 2)/(σ_β^2) + (X_prop_1'*X_prop_1)/(σ_proposed_1^2))
  β_var_prop_1 = 0.5*(β_var_prop_1 + β_var_prop_1')
  β_mean_prop_1 = β_var_prop_1*((X_prop_1'*y_star_1)/(σ_proposed_1^2))
  β_proposed_1 = rand(MvNormal(β_mean_prop_1, β_var_prop_1), 1)
  res_var_prop_1 = sum((y_star_1 - X_prop_1*β_proposed_1).^2)

  log_likelik_1_proposed = log_likelihood_segment(β_proposed_1, ω_star_1, a, b, σ_proposed_1)


  # --- Proposing beta for new partition 2
  a = floor(Int64, s_proposed[index_s + 1])
  if (s_proposed[index_s + 2] == n)
    b = floor(Int64, s_proposed[index_s + 2])
  else
    b = floor(Int64, s_proposed[index_s + 2]) - 1
  end

  y_star_2 = y[a:b]
  n_obs_prop_2 = length(y_star_2)
  ω_star_2 = copy(ω_current)
  n_freq_prop_2 = length(ω_star_2)

  X_prop_2 = get_X(ω_star_2, a, b)
  β_var_prop_2 = inv(eye(2*n_freq_prop_2 + 2)/(σ_β^2) + (X_prop_2'*X_prop_2)/(σ_proposed_2^2))
  β_var_prop_2 = 0.5*(β_var_prop_2 + β_var_prop_2')
  β_mean_prop_2 = β_var_prop_2*((X_prop_2'*y_star_2)/(σ_proposed_2^2))
  β_proposed_2 = rand(MvNormal(β_mean_prop_2, β_var_prop_2), 1)
  res_var_prop_2 = sum((y_star_2 - X_prop_2*β_proposed_2).^2)


  log_likelik_2_proposed = log_likelihood_segment(β_proposed_2, ω_star_2, a, b, σ_proposed_2)



  # ---- Auxiliary for proposal ratio β

  a = floor(Int64, s_current[index_s])
  if (s_current[index_s+1]==n)
    b = floor(Int64, s_current[index_s+1])
  else
    b = floor(Int64, s_current[index_s+1]) - 1
  end
  y_curr = y[a:b]

  X_curr = get_X(ω_current, a, b)
  β_var_curr = inv(eye(2*n_freq_current+2)/(σ_β^2) + (X_curr'*X_curr)/(σ_current^2))
  β_var_curr = (β_var_curr + β_var_curr')/2
  β_mean_curr = β_var_curr*((X_curr'*y_curr)/(σ_current^2))


  ########## ------- Evaluating acceptance probability  ------- ##########

  # ------------ Log_likelik_ratio

  log_likelik_proposed = log_likelik_1_proposed + log_likelik_2_proposed
  log_likelik_ratio = log_likelik_proposed - log_likelik_current

  # ------------ Prior_ratio

  # - n_CP
  log_prior_n_CP_ratio = log(pdf(Poisson(λ_NS), n_CP_current + 1)/
                    pdf(Poisson(λ_NS), n_CP_current))

  # - s
  log_prior_s_ratio = log(((2*(n_CP_current + 1)*(n_CP_current*2 + 3))/((n-1)^2))*
                            ((s_proposed[index_s+2] - s_proposed[index_s+1])*
                             (s_proposed[index_s+1] - s_proposed[index_s])/
                             (s_proposed[index_s+2] - s_proposed[index_s])
                            ))

  # Prior additional number of frequencies
  log_prior_n_freq = log(pdf(Poisson(λ_S), n_freq_current))

  # Prior additional frequencies
  log_prior_freq = log(2^(n_freq_current))
  # - beta
  log_prior_β_1_proposed = log(pdf(MvNormal(zeros(length(β_proposed_1)), (σ_β^2)*eye(length(β_proposed_1))),
                            β_proposed_1)[1])

  log_prior_β_2_proposed = log(pdf(MvNormal(zeros(length(β_proposed_2)), (σ_β^2)*eye(length(β_proposed_2))),
                            β_proposed_2)[1])
  log_prior_β_proposed = log_prior_β_1_proposed + log_prior_β_2_proposed

  log_prior_β_current = log(pdf(MvNormal(zeros(length(β_current)), (σ_β^2)*eye(length(β_current))),
                           β_current)[1])
  log_prior_β_ratio = log_prior_β_proposed - log_prior_β_current

  # σ
  log_prior_σ2_current = log(pdf(InverseGamma(ν0/2, γ0/2), σ_current^2))
  log_prior_σ2_proposed_1 = log(pdf(InverseGamma(ν0/2, γ0/2), σ_proposed_1^2))
  log_prior_σ2_proposed_2 = log(pdf(InverseGamma(ν0/2, γ0/2), σ_proposed_2^2))
  log_prior_σ2 = (log_prior_σ2_proposed_1 + log_prior_σ2_proposed_2) -
                 log_prior_σ2_current

  log_prior_ratio = log_prior_n_CP_ratio + log_prior_s_ratio +
                    log_prior_n_freq + log_prior_freq + log_prior_β_ratio +
                    log_prior_σ2


  # ----------- Proposal_ratio

  death_prob = c_NS*min(1, pdf(Poisson(λ_NS), n_CP_current)/
                 pdf(Poisson(λ_NS), n_CP_current+1))
  birth_prob = c_NS*min(1, pdf(Poisson(λ_NS), n_CP_current+1)/
                 pdf(Poisson(λ_NS), n_CP_current))

  log_proposal_ratio_n_CP = log(death_prob) - log(birth_prob)
  log_proposal_ratio_s = log(length_support/(n_CP_current + 1))
  log_proposal_ratio_m_omega = log(1/2)


  log_proposal_β_current = log(pdf(MvNormal(β_mean_curr, β_var_curr), β_current))

  log_proposal_β_1_proposed = log.(pdf(MvNormal(β_mean_prop_1, β_var_prop_1), β_proposed_1))[1]
  log_proposal_β_2_proposed = log.(pdf(MvNormal(β_mean_prop_2, β_var_prop_2), β_proposed_2))[1]

  log_proposal_β_ratio = log_proposal_β_current -
                        (log_proposal_β_1_proposed + log_proposal_β_2_proposed)


  log_proposal_ratio = log_proposal_ratio_n_CP + log_proposal_ratio_m_omega +
                       log_proposal_ratio_s + log_proposal_β_ratio


  # --- Jacobian (for σ)
  log_jacobian = log((2*(σ_proposed_1 + σ_proposed_2)^2))


  # ---------- Output proposed objects
  if (index_s == 1)

    # β
    β_aux_1 = β_seg[(index_s+1):(n_CP_current+1), :]
    β_aux_2 = reshape(vcat(vec(β_proposed_1), zeros((2*n_freq_max+2) - length(β_proposed_1))), 1, (2*n_freq_max+2))
    β_aux_3 = reshape(vcat(vec(β_proposed_2), zeros((2*n_freq_max+2) - length(β_proposed_2))), 1, (2*n_freq_max+2))
    β_aux = [β_aux_2; β_aux_3; β_aux_1]
    β_zeros_aux = zeros((n_CP_max - size(β_aux)[1] + 1), (2*n_freq_max+2))
    β_proposed = [β_aux; β_zeros_aux]


    # ω
    ω_aux_1 = ω_seg[(index_s+1):(n_CP_current+1), :]
    ω_aux_2 = reshape(vcat(vec(ω_star_1), zeros(n_freq_max - length(ω_star_1))), 1, n_freq_max)
    ω_aux_3 = reshape(vcat(vec(ω_star_2), zeros(n_freq_max - length(ω_star_2))), 1, n_freq_max)
    ω_aux =  [ω_aux_2; ω_aux_3; ω_aux_1;]
    ω_zeros_aux = zeros((n_CP_max - size(ω_aux)[1] + 1), n_freq_max)
    ω_proposed =[ω_aux; ω_zeros_aux]

    # n_freq
    n_freq_aux_1 = n_freq_seg[(index_s+1):(n_CP_current+1)]
    n_freq_aux_2 = vcat(n_freq_prop_1, n_freq_prop_2)
    n_freq_aux = [n_freq_aux_2; n_freq_aux_1]
    n_freq_zeros_aux = zeros(Int64, (n_CP_max - size(n_freq_aux)[1] + 1))
    n_freq_proposed = [n_freq_aux; n_freq_zeros_aux]

    # σ
    σ_aux_1  = σ_seg[(index_s+1):(n_CP_current+1)]
    σ_aux_2 = vcat(σ_proposed_1, σ_proposed_2)
    σ_aux = [σ_aux_2; σ_aux_1]
    σ_zeros_aux = zeros(Float64, (n_CP_max - size(σ_aux)[1] + 1))
    σ_proposed = [σ_aux; σ_zeros_aux]

    # log_likelik
    log_likelik_aux_1  = log_likelik_seg[(index_s+1):(n_CP_current+1)]
    log_likelik_aux_2 = vcat(log_likelik_1_proposed, log_likelik_2_proposed )
    log_likelik_aux = [log_likelik_aux_2; log_likelik_aux_1;]
    log_likelik_zeros_aux = zeros(Int64, (n_CP_max - size(log_likelik_aux)[1] + 1))
    log_likelik_seg_proposed = [log_likelik_aux; log_likelik_zeros_aux]

  elseif (index_s == (n_CP_current + 1))

    # β
    β_aux_1 = β_seg[1:(index_s-1), :]
    β_aux_2 = reshape(vcat(vec(β_proposed_1), zeros((2*n_freq_max+2) - length(β_proposed_1))), 1, (2*n_freq_max+2))
    β_aux_3 = reshape(vcat(vec(β_proposed_2), zeros((2*n_freq_max+2) - length(β_proposed_2))), 1, (2*n_freq_max+2))
    β_aux = [β_aux_1; β_aux_2; β_aux_3]
    β_zeros_aux = zeros((n_CP_max - size(β_aux)[1] + 1), (2*n_freq_max+2))
    β_proposed = [β_aux; β_zeros_aux]


    # ω
    ω_aux_1 = ω_seg[1:(index_s-1), :]
    ω_aux_2 = reshape(vcat(vec(ω_star_1), zeros(n_freq_max - length(ω_star_1))), 1, n_freq_max)
    ω_aux_3 = reshape(vcat(vec(ω_star_2), zeros(n_freq_max - length(ω_star_2))), 1, n_freq_max)
    ω_aux = [ω_aux_1; ω_aux_2; ω_aux_3;]
    ω_zeros_aux = zeros((n_CP_max - size(ω_aux)[1] + 1), n_freq_max)
    ω_proposed =[ω_aux; ω_zeros_aux]

    # n_freq
    n_freq_aux_1 = n_freq_seg[1:(index_s-1)]
    n_freq_aux_2 = vcat(n_freq_prop_1, n_freq_prop_2)
    n_freq_aux = [n_freq_aux_1; n_freq_aux_2]
    n_freq_zeros_aux = zeros(Int64, (n_CP_max - size(n_freq_aux)[1] + 1))
    n_freq_proposed = [n_freq_aux; n_freq_zeros_aux]

    # σ
    σ_aux_1 = σ_seg[1:(index_s-1)]
    σ_aux_2 = vcat(σ_proposed_1, σ_proposed_2)
    σ_aux = [σ_aux_1; σ_aux_2]
    σ_zeros_aux = zeros(Float64, (n_CP_max - size(σ_aux)[1] + 1))
    σ_proposed = [σ_aux; σ_zeros_aux]

    # log_likelik
    log_likelik_aux_1  = log_likelik_seg[1:(index_s-1)]
    log_likelik_aux_2 = vcat(log_likelik_1_proposed, log_likelik_2_proposed )
    log_likelik_aux = [log_likelik_aux_1; log_likelik_aux_2]
    log_likelik_zeros_aux = zeros(Int64, (n_CP_max - size(log_likelik_aux)[1] + 1))
    log_likelik_seg_proposed = [log_likelik_aux; log_likelik_zeros_aux]

  else

    # β
    β_aux_1 = β_seg[1:(index_s-1), :]
    β_aux_2 = reshape(vcat(vec(β_proposed_1), zeros((2*n_freq_max+2) - length(β_proposed_1))), 1, (2*n_freq_max+2))
    β_aux_3 = reshape(vcat(vec(β_proposed_2), zeros((2*n_freq_max+2) - length(β_proposed_2))), 1, (2*n_freq_max+2))
    β_aux_4 = β_seg[(index_s+1:(n_CP_current+1)), :]
    β_aux = [β_aux_1; β_aux_2; β_aux_3; β_aux_4;]
    β_zeros_aux = zeros((n_CP_max - size(β_aux)[1] + 1), (2*n_freq_max+2))
    β_proposed = [β_aux; β_zeros_aux]


    # ω
    ω_aux_1 = ω_seg[1:(index_s-1), :]
    ω_aux_2 = reshape(vcat(vec(ω_star_1), zeros(n_freq_max - length(ω_star_1))), 1, n_freq_max)
    ω_aux_3 = reshape(vcat(vec(ω_star_2), zeros(n_freq_max - length(ω_star_2))), 1, n_freq_max)
    ω_aux_4 = ω_seg[(index_s+1:(n_CP_current+1)), :]
    ω_aux = [ω_aux_1; ω_aux_2; ω_aux_3; ω_aux_4; ]
    ω_zeros_aux = zeros((n_CP_max - size(ω_aux)[1] + 1), n_freq_max)
    ω_proposed =[ω_aux; ω_zeros_aux]

    # n_freq
    n_freq_aux_1 = n_freq_seg[1:(index_s-1)]
    n_freq_aux_2 = vcat(n_freq_prop_1, n_freq_prop_2)
    n_freq_aux_3 = n_freq_seg[(index_s+1:(n_CP_current+1))]
    n_freq_aux = [n_freq_aux_1; n_freq_aux_2; n_freq_aux_3]
    n_freq_zeros_aux = zeros(Int64, (n_CP_max - size(n_freq_aux)[1] + 1))
    n_freq_proposed = [n_freq_aux; n_freq_zeros_aux]

    # σ
    σ_aux_1 = σ_seg[1:(index_s-1)]
    σ_aux_2 = vcat(σ_proposed_1, σ_proposed_2)
    σ_aux_3 = σ_seg[(index_s+1:(n_CP_current+1))]
    σ_aux = [σ_aux_1; σ_aux_2; σ_aux_3]
    σ_zeros_aux = zeros(Float64, (n_CP_max - size(σ_aux)[1] + 1))
    σ_proposed = [σ_aux; σ_zeros_aux]

    # log_likelik
    log_likelik_aux_1  = log_likelik_seg[1:(index_s-1)]
    log_likelik_aux_2 = vcat(log_likelik_1_proposed, log_likelik_2_proposed )
    log_likelik_aux_3 = log_likelik_seg[(index_s+1:(n_CP_current+1))]
    log_likelik_aux = [log_likelik_aux_1; log_likelik_aux_2; log_likelik_aux_3]
    log_likelik_zeros_aux = zeros(Int64, (n_CP_max - size(log_likelik_aux)[1] + 1))
    log_likelik_seg_proposed = [log_likelik_aux; log_likelik_zeros_aux]

  end


  # Accept / Reject

  MH_ratio = log_likelik_ratio + log_prior_ratio + log_proposal_ratio + log_jacobian
  epsilon = min(1, exp(MH_ratio))
  U = rand()

  # ---- # Accept / Reject

  if (U <= epsilon)
    n_CP_sample[t] = n_CP_current + 1
    s_sample[1:n_CP_sample[t], t] = s_proposed[2:(end-1)]
    β_sample[:, :, t] = β_proposed
    ω_sample[:, :, t] = ω_proposed
    n_freq_sample[:, t] = n_freq_proposed
    σ_sample[:, t] = σ_proposed
    log_likelik_sample[:, t] = log_likelik_seg_proposed
    accepted = true
  else
    n_CP_sample[t] = n_CP_current
    if (n_CP_current == 0)
      s_sample[1, t] = 0.0
    else
      s_sample[1:n_CP_sample[t], t] = s_current[2:(end-1)]
    end
    β_sample[:, :, t] = β_seg
    ω_sample[:, :, t] = ω_seg
    σ_sample[1:(n_CP_sample[t] + 1), t] = σ_seg
    n_freq_sample[1:(n_CP_sample[t] + 1), t] = n_freq_seg
    log_likelik_sample[1:(n_CP_sample[t] + 1), t] = log_likelik_seg
    accepted = false
  end


  if (accepted == true)
    if (index_s == 1)

      index_seg_update = collect((index_s + 1):(n_CP_current + 1))
      index_seg_adjacent = [index_s, index_s+1]

      s_aux = vcat(1, s_sample[1:n_CP_sample[t], t], n_obs)

      # --- Updating untouched segments
      for j in index_seg_update
        a = floor(Int64, s_aux[j+1])
        if (j == n_CP_current + 1)
          b = floor(Int64, n_obs)
        else
          b = floor(Int64, (s_aux[j+2] - 1))
        end
        global y = data[a:b]
        global n = length(y)
        global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

        m_current = n_freq_sample[j, t-1]
        σ_current = σ_sample[j, t-1]
        β_current = β_sample[j, 1:(2*m_current+2), t-1]
        ω_current = ω_sample[j, 1:(m_current), t-1]

        MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
        n_freq_sample[j+1, t] = MCMC_S["m"]
        β_sample[j+1, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2) - length(MCMC_S["β"])))
        ω_sample[j+1, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
        σ_sample[j+1, t] = MCMC_S["σ"]
        log_likelik_sample[j+1, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

      end

      for j in index_seg_adjacent

        a = floor(Int64, s_aux[j])
        if (j == n_CP_current + 1)
          b = floor(Int64, n_obs)
        else
          b = floor(Int64, (s_aux[j+1] - 1))
        end

        global y = data[a:b]
        global n = length(y)
        global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

        m_current = n_freq_sample[j, t]
        β_current = β_sample[j, 1:(2*m_current+2), t]
        ω_current = ω_sample[j, 1:(m_current), t]
        σ_current = σ_sample[j, t]

        MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
        n_freq_sample[j, t] = MCMC_S["m"]
        β_sample[j, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
        ω_sample[j, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
        σ_sample[j, t] = MCMC_S["σ"]
        log_likelik_sample[j, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

      end

    elseif (index_s == (n_CP_current + 1))
      s_aux = vcat(1, s_sample[1:n_CP_sample[t], t], n_obs)
      index_seg_update = collect(1:(index_s - 1))
      index_seg_adjacent = [index_s, index_s+1]

      for j in index_seg_update

        a = floor(Int64, s_aux[j])
        if (j == n_CP_current + 1)
          b = floor(Int64, n_obs)
        else
          b = floor(Int64, (s_aux[j+1] - 1))
        end
        global y = data[a:b]
        global n = length(y)
        global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

        m_current = n_freq_sample[j, t-1]
        σ_current = σ_sample[j, t-1]
        β_current = β_sample[j, 1:(2*m_current+2), t-1]
        ω_current = ω_sample[j, 1:(m_current), t-1]

        MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
        n_freq_sample[j, t] = MCMC_S["m"]
        σ_sample[j, t] = MCMC_S["σ"]
        β_sample[j, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
        ω_sample[j, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
        log_likelik_sample[j, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

      end

      for j in index_seg_adjacent
        a = floor(Int64, s_aux[j])
        if (j == n_CP_current + 2)
          b = floor(Int64, n_obs)
        else
          b = floor(Int64, (s_aux[j+1] - 1))
        end
        global y = data[a:b]
        global n = length(y)
        global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

        m_current = n_freq_sample[j, t]
        σ_current = σ_sample[j, t]
        β_current = β_sample[j, 1:(2*m_current+2), t]
        ω_current = ω_sample[j, 1:(m_current), t]

        MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
        n_freq_sample[j, t] = MCMC_S["m"]
        σ_sample[j, t] = MCMC_S["σ"]
        β_sample[j, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
        ω_sample[j, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
        log_likelik_sample[j, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])


      end
    else
      s_aux = vcat(1, s_sample[1:n_CP_sample[t], t], n_obs)
      index_seg_update = vcat(collect(1:(index_s - 1)), collect((index_s+2):(n_CP_current+2)))
      index_seg_adjacent = [index_s, index_s+1]


      for j in index_seg_update
        if (j < index_s)
          a = floor(Int64, s_aux[j])
          if (j == n_CP_current + 1)
            b = floor(Int64, n_obs)
          else
            b = floor(Int64, (s_aux[j+1] - 1))
          end
          global y = data[a:b]
          global n = length(y)
          global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

          m_current = n_freq_sample[j, t-1]
          σ_current = σ_sample[j, t-1]
          β_current = β_sample[j, 1:(2*m_current+2), t-1]
          ω_current = ω_sample[j, 1:(m_current), t-1]

          MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
          n_freq_sample[j, t] = MCMC_S["m"]
          σ_sample[j, t] = MCMC_S["σ"]
          β_sample[j, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
          ω_sample[j, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
          log_likelik_sample[j, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])


        else
          a = floor(Int64, s_aux[j])
          if (j == n_CP_current + 2)
            b = floor(Int64, n_obs)
          else
            b = floor(Int64, (s_aux[j+1] - 1))
          end
          global y = data[a:b]
          global n = length(y)
          global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

          m_current = n_freq_sample[j-1, t-1]
          σ_current = σ_sample[j-1, t-1]
          β_current = β_sample[j-1, 1:(2*m_current+2), t-1]
          ω_current = ω_sample[j-1, 1:(m_current), t-1]


          MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
          n_freq_sample[j, t] = MCMC_S["m"]
          σ_sample[j, t] = MCMC_S["σ"]
          β_sample[j, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
          ω_sample[j, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
          log_likelik_sample[j, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

        end
      end

      for j in index_seg_adjacent

        a = floor(Int64, s_aux[j])
        if (j == n_CP_current + 2)
          b = floor(Int64, n_obs)
        else
          b = floor(Int64, (s_aux[j+1] - 1))
        end
        global y = data[a:b]
        global n = length(y)
        global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

        m_current = n_freq_sample[j, t]
        σ_current = σ_sample[j, t]
        β_current = β_sample[j, 1:(2*m_current+2), t]
        ω_current = ω_sample[j, 1:(m_current), t]

        MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
        n_freq_sample[j, t] = MCMC_S["m"]
        σ_sample[j, t] = MCMC_S["σ"]
        β_sample[j, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
        ω_sample[j, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
        log_likelik_sample[j, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

      end
    end
  else
    s_aux = vcat(1, s_sample[1:n_CP_current, t], n_obs)
    for j in 1:(n_CP_current + 1)
      a = floor(Int64, s_aux[j])
      if (j == n_CP_current + 1)
        b = floor(Int64, n_obs)
      else
        b = floor(Int64, (s_aux[j+1] - 1))
      end
      global y = data[a:b]
      global n = length(y)
      global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

      m_current = n_freq_sample[j, t-1]
      σ_current = σ_sample[j, t-1]
      β_current = β_sample[j, 1:(2*m_current+2), t-1]
      ω_current = ω_sample[j, 1:(m_current), t-1]

      MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
      n_freq_sample[j, t] = MCMC_S["m"]
      σ_sample[j,t] = MCMC_S["σ"]
      β_sample[j, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
      ω_sample[j, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
      log_likelik_sample[j, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])
    end
  end


  output = Dict("n_CP" => n_CP_sample, "β" => β_sample, "σ" => σ_sample,
                "ω" => ω_sample, "s" => s_sample, "n_freq" => n_freq_sample,
                "log_likelik" => log_likelik_sample)

  return output
end


# Function: non-stationary model, death step.

function death_move_non_stationary(t, n_CP_sample, β_sample, ω_sample,
                          σ_sample, s_sample, n_freq_sample, log_likelik_sample, hyperparms)


  λ_NS = hyperparms[:λ_NS] # poisson prior parameter, i.e n of cp ∼ Poisson(λ_NS)
  c_NS = hyperparms[:c_NS]  # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
  ψ_NS =hyperparms[:ψ_NS] # minumum distance between change-points 
  σ_β = hyperparms[:σ_β]  
  σ_RW_s = hyperparms[:σ_RW_s]   
  δ_s_mixing = hyperparms[:δ_s_mixing] 
  n_CP_max = hyperparms[:n_CP_max] # maximum number of change-points
  n_freq_max = hyperparms[:n_freq_max] # maximum number of frequencies in each segment. 
  ν0 = hyperparms[:ν0] # prior σ², InverseGamma(ν0/2, η0/2)
  γ0 = hyperparms[:γ0] # prior σ², InverseGamma(ν0/2, η0/2)     
  λ_S = hyperparms[:λ_S] # poisson prior parameter, i.e n of freq ∼ Poisson(λ_S)
  c_S = hyperparms[:c_S] # constant for birth/death prbability, c ∈ (0, 0.5) -- Equation (6)
  ϕ_ω = hyperparms[:ϕ_ω] # birth step, frequency is sampled from Unif(0, ψ_ω)
  ψ_ω = hyperparms[:ψ_ω] # miminum distance between frequencies
  σ_RW_ω_hyper = hyperparms[:σ_RW_ω_hyper]  # variance parameter for random walk when relocating a freq :
              

  global y = data
  global n = length(data)

  n_CP_current = n_CP_sample[t-1]
  β_seg = β_sample[:, :, t-1]
  ω_seg = ω_sample[:, :, t-1]
  σ_seg = σ_sample[1:n_CP_current+1, t-1]
  s = s_sample[1:n_CP_current, t-1]
  n_freq_seg = n_freq_sample[1:(n_CP_current +1), t-1]
  log_likelik_seg = log_likelik_sample[1:(n_CP_current +1), t-1]

  s_current = vcat(1, s, n)
  index_s = sample(1:n_CP_current) # candidate to die
  s_proposed = vcat(s_current[1:(index_s)], s_current[(index_s+2):end])

  # Evaluating the support of s, in the scenario that s_current[index+1] were delated
  support_s = Array{Vector{Float64}}(undef, n_CP_current)
  for k in 1:(n_CP_current)
    support_s[k] = [s_proposed[k]+ψ_NS, s_proposed[k+1]-ψ_NS]
  end


  # Notice that it's possible that [s_proposed[k]+phi, s_proposed[k+1]-phi]
  # could not make sense in terms that the interval[a, b], a>b.
  # This is solved by the function get_support_s
  length_support = get_support_s(support_s)

  ω_current_1 = ω_seg[index_s, 1:n_freq_seg[index_s]]
  n_freq_current_1 = length(ω_current_1)
  σ_current_1 = σ_seg[index_s]

  ω_current_2 = ω_seg[index_s+1, 1:n_freq_seg[index_s+1]]
  n_freq_current_2 = length(ω_current_2)
  σ_current_2 = σ_seg[index_s+1]

  β_current_1 = β_seg[index_s, 1:(2*n_freq_seg[index_s]+2)]
  β_current_2 = β_seg[index_s+1, 1:(2*n_freq_seg[index_s+1]+2)]

  log_likelik_1_current = log_likelik_seg[index_s]
  log_likelik_2_current = log_likelik_seg[index_s + 1]


  wprobs = [0.5, 0.5]
  items = [ω_current_1, ω_current_2]
  # ω_star = sort(vcat(ω_current_1, ω_current_2))
  ω_star = sample(items, Weights(wprobs))
  n_freq_star = length(ω_star)

  # Proposing σ
  σ_star = sqrt(sqrt((σ_current_1^2)*(σ_current_2^2)))


  # Proposing β
  a = floor(Int64, s_proposed[index_s])
  if (s_proposed[index_s + 1] == n)
    b = floor(Int64, s_proposed[index_s + 1])
  else
    b = floor(Int64, s_proposed[index_s + 1]) - 1
  end

  y_star = y[a:b]
  X_prop = get_X(ω_star, a, b)
  β_var_prop = inv(eye(2*n_freq_star + 2)/(σ_β^2) + (X_prop'*X_prop)/(σ_star^2))
  β_var_prop = 0.5*(β_var_prop + β_var_prop')
  β_mean_prop = β_var_prop*((X_prop'*y_star)/(σ_star^2))

  β_proposed_star = rand(MvNormal(β_mean_prop, β_var_prop), 1)
  log_likelik_star_proposed = log_likelihood_segment(β_proposed_star,
                                                     ω_star, a, b, σ_star)

  # ----------- Auxiliary objects for proposal ratio:

  a = floor(Int64, s_current[index_s])
  b = floor(Int64, s_current[index_s+1]) - 1
  y_curr_1 = y[a:b]


  X_curr_1 = get_X(ω_current_1, a, b)
  β_var_curr_1 = inv(eye(2*n_freq_current_1 + 2)/(σ_β^2) + (X_curr_1'*X_curr_1)/(σ_current_1^2))
  β_var_curr_1 = 0.5*(β_var_curr_1 + β_var_curr_1')
  β_mean_curr_1 = β_var_curr_1*((X_curr_1'*y_curr_1)/(σ_current_1^2))


  a = floor(Int64, s_current[index_s+1])
  if (s_current[index_s+2]==n)
    b = floor(Int64, s_current[index_s+2])
  else
    b = floor(Int64, s_current[index_s+2]) - 1
  end

  y_curr_2 = y[a:b]

  X_curr_2 = get_X(ω_current_2, a, b)
  β_var_curr_2 = inv(eye(2*n_freq_current_2 + 2)/(σ_β^2) + (X_curr_2'*X_curr_2)/(σ_current_2^2))
  β_var_curr_2 = 0.5*(β_var_curr_2 + β_var_curr_2')
  β_mean_curr_2 = β_var_curr_2*((X_curr_2'*y_curr_2)/(σ_current_2^2))



  # ------------ Log_likelik_ratio
  log_likelik_proposed = log_likelik_star_proposed
  log_likelik_current = log_likelik_1_current + log_likelik_2_current
  log_likelik_ratio = log_likelik_proposed - log_likelik_current


  # ------------ Prior_ratio

  # - n_CP
  log_prior_n_CP_ratio = log(pdf(Poisson(λ_NS), n_CP_current-1)/
                    pdf(Poisson(λ_NS), n_CP_current))

  # - s
  log_prior_s_ratio = log((((n-1)^2)/(2*(n_CP_current)*((n_CP_current - 1)*2 + 3)))*
        (s_current[index_s+2] - s_current[index_s])/
        ((s_current[index_s+2] - s_current[index_s+1])*(s_current[index_s+1] - s_current[index_s])))

  # n_freq
  log_prior_n_freq = log(pdf(Poisson(λ_S), n_freq_star)) - log(pdf(Poisson(λ_S), n_freq_current_1)) -
                           log(pdf(Poisson(λ_S), n_freq_current_2))

   # freq
  log_prior_freq = log(2^(n_freq_star)) - log(2^(n_freq_current_1)) - log(2^(n_freq_current_2))


  # - β
  log_prior_β_proposed = log(pdf(MvNormal(zeros(length(β_proposed_star)), (σ_β^2)*eye(length(β_proposed_star))),
                            β_proposed_star)[1])

  log_prior_β_1_current = log(pdf(MvNormal(zeros(length(β_current_1)), (σ_β^2)*eye(length(β_current_1))),
                            β_current_1)[1])
  log_prior_β_2_current = log(pdf(MvNormal(zeros(length(β_current_2)), (σ_β^2)*eye(length(β_current_2))),
                            β_current_2)[1])

  log_prior_β_ratio = log_prior_β_proposed - (log_prior_β_1_current + log_prior_β_2_current)


  # - σ
  log_prior_σ2_proposed = log(pdf(InverseGamma(ν0/2, γ0/2), σ_star^2))

  log_prior_σ2_1_current = log(pdf(InverseGamma(ν0/2, γ0/2), σ_current_1^2))
  log_prior_σ2_2_current = log(pdf(InverseGamma(ν0/2, γ0/2), σ_current_2^2))

  log_prior_σ2_ratio = log_prior_σ2_proposed - (log_prior_σ2_1_current + log_prior_σ2_2_current)

  log_prior_ratio = log_prior_n_CP_ratio + log_prior_n_freq + log_prior_freq +
                   log_prior_s_ratio + log_prior_σ2_ratio + log_prior_β_ratio


  # ----------- Proposal_ratio
  death_prob = c_NS*min(1, pdf(Poisson(λ_NS), n_CP_current-1)/
                 pdf(Poisson(λ_NS), n_CP_current))
  birth_prob = c_NS*min(1, pdf(Poisson(λ_NS), n_CP_current)/
                 pdf(Poisson(λ_NS), n_CP_current-1))

  log_proposal_ratio_n_CP = log(birth_prob) - log(death_prob)

  log_proposal_ratio_s = log(n_CP_current/length_support)

  log_proposal_ratio_m_omega = -log(0.5)


  log_proposal_β_1_current = log(pdf(MvNormal(β_mean_curr_1, β_var_curr_1), β_current_1)[1])
  log_proposal_β_2_current = log(pdf(MvNormal(β_mean_curr_2, β_var_curr_2), β_current_2)[1])

  log_proposal_β_proposed = log(pdf(MvNormal(β_mean_prop, β_var_prop), β_proposed_star)[1])

  log_proposal_β_ratio = (log_proposal_β_1_current + log_proposal_β_2_current) - log_proposal_β_proposed

  log_proposal_ratio = log_proposal_ratio_n_CP + log_proposal_ratio_s + log_proposal_β_ratio +
                       log_proposal_ratio_m_omega


  # Jacobian
  log_jacobian = -log(2*(σ_current_1 + σ_current_2)^2)


  if (index_s == 1)

    # β
    β_aux_1 = β_seg[(index_s+2):(n_CP_current + 1), :]
    β_aux_2 = reshape(vcat(vec(β_proposed_star), zeros((2*n_freq_max+2) - length(β_proposed_star))), 1, (2*n_freq_max+2))
    β_aux = [β_aux_2; β_aux_1]
    β_zeros_aux = zeros((n_CP_max - size(β_aux)[1] + 1),(2*n_freq_max+2))
    β_proposed = [β_aux; β_zeros_aux]

    # ω
    ω_aux_1 = ω_seg[(index_s+2):(n_CP_current + 1), :]
    ω_aux_2 = reshape(vcat(vec(ω_star), zeros(n_freq_max - length(ω_star))), 1, n_freq_max)
    ω_aux = [ω_aux_2; ω_aux_1]
    ω_zeros_aux = zeros((n_CP_max - size(ω_aux)[1] + 1), n_freq_max)
    ω_proposed =[ω_aux; ω_zeros_aux]

    # n_freq
    n_freq_aux_1 = n_freq_seg[(index_s+2):(n_CP_current + 1)]
    n_freq_aux_2 = n_freq_star
    n_freq_aux = [n_freq_aux_2; n_freq_aux_1]
    n_freq_zeros_aux = zeros(Int64, (n_CP_max - size(n_freq_aux)[1] + 1))
    n_freq_proposed = [n_freq_aux; n_freq_zeros_aux]

    # σ
    σ_aux_1 = σ_seg[(index_s+2):(n_CP_current + 1)]
    σ_aux_2 = σ_star
    σ_aux = [σ_aux_2; σ_aux_1]
    σ_zeros_aux = zeros(Float64, (n_CP_max - size(σ_aux)[1] + 1))
    σ_proposed = [σ_aux; σ_zeros_aux]


    # log_likelik
    log_likelik_aux_1  = log_likelik_seg[(index_s+2):(n_CP_current + 1)]
    log_likelik_aux_2 = log_likelik_star_proposed
    log_likelik_aux = [log_likelik_aux_2; log_likelik_aux_1]
    log_likelik_zeros_aux = zeros(Int64, (n_CP_max - size(log_likelik_aux)[1] + 1))
    log_likelik_seg_proposed = [log_likelik_aux; log_likelik_zeros_aux ]
  elseif (index_s == n_CP_current)

    # β
    β_aux_1 = β_seg[1:(index_s-1), :]
    β_aux_2 = reshape(vcat(vec(β_proposed_star), zeros((2*n_freq_max+2) - length(β_proposed_star))), 1, (2*n_freq_max+2))
    β_aux = [β_aux_1; β_aux_2]
    β_zeros_aux = zeros((n_CP_max - size(β_aux)[1] + 1), (2*n_freq_max+2))
    β_proposed = [β_aux; β_zeros_aux]

    # ω
    ω_aux_1 = ω_seg[1:(index_s-1), :]
    ω_aux_2 = reshape(vcat(vec(ω_star), zeros(n_freq_max - length(ω_star))), 1, n_freq_max)
    ω_aux = [ω_aux_1; ω_aux_2]
    ω_zeros_aux = zeros((n_CP_max - size(ω_aux)[1] + 1), n_freq_max)
    ω_proposed =[ω_aux; ω_zeros_aux]

    # n_freq
    n_freq_aux_1 = n_freq_seg[1:(index_s-1)]
    n_freq_aux_2 = n_freq_star
    n_freq_aux = [n_freq_aux_1; n_freq_aux_2]
    n_freq_zeros_aux = zeros(Int64, (n_CP_max - size(n_freq_aux)[1] + 1))
    n_freq_proposed = [n_freq_aux; n_freq_zeros_aux]

    # σ
    σ_aux_1 = σ_seg[1:(index_s-1)]
    σ_aux_2 = σ_star
    σ_aux = [σ_aux_1; σ_aux_2]
    σ_zeros_aux = zeros(Float64, (n_CP_max - size(σ_aux)[1] + 1))
    σ_proposed = [σ_aux; σ_zeros_aux]

    # log_likelik
    log_likelik_aux_1  = log_likelik_seg[1:(index_s-1)]
    log_likelik_aux_2 = log_likelik_star_proposed
    log_likelik_aux = [log_likelik_aux_1; log_likelik_aux_2]
    log_likelik_zeros_aux = zeros(Int64, (n_CP_max - size(log_likelik_aux)[1] + 1))
    log_likelik_seg_proposed = [log_likelik_aux; log_likelik_zeros_aux ]
  else

    # β
    β_aux_1 = β_seg[(1:(index_s - 1)), :]
    β_aux_2 = reshape(vcat(vec(β_proposed_star), zeros((2*n_freq_max+2) - length(β_proposed_star))), 1, (2*n_freq_max+2))
    β_aux_3 = β_seg[(index_s+2:(n_CP_current+1)), :]
    β_aux = [β_aux_1; β_aux_2; β_aux_3]
    β_zeros_aux = zeros((n_CP_max - size(β_aux)[1] + 1), (2*n_freq_max+2))
    β_proposed = [β_aux; β_zeros_aux]

    # ω
    ω_aux_1 = ω_seg[1:(index_s-1), :]
    ω_aux_2 = reshape(vcat(vec(ω_star), zeros(n_freq_max - length(ω_star))), 1, n_freq_max)
    ω_aux_3 = ω_seg[(index_s+2:(n_CP_current+1)), :]
    ω_aux = [ω_aux_1; ω_aux_2; ω_aux_3]
    ω_zeros_aux = zeros((n_CP_max - size(ω_aux)[1] + 1), n_freq_max)
    ω_proposed =[ω_aux; ω_zeros_aux]

    # n_freq
    n_freq_aux_1 = n_freq_seg[1:(index_s-1)]
    n_freq_aux_2 = n_freq_star
    n_freq_aux_3 = n_freq_seg[(index_s+2:(n_CP_current+1))]
    n_freq_aux = [n_freq_aux_1; n_freq_aux_2; n_freq_aux_3]
    n_freq_zeros_aux = zeros(Int64, (n_CP_max - size(n_freq_aux)[1] + 1))
    n_freq_proposed = [n_freq_aux; n_freq_zeros_aux]

    # σ
    σ_aux_1 = σ_seg[1:(index_s-1)]
    σ_aux_2 = σ_star
    σ_aux_3 = σ_seg[(index_s+2:(n_CP_current+1))]
    σ_aux = [σ_aux_1; σ_aux_2; σ_aux_3]
    σ_zeros_aux = zeros(Float64, (n_CP_max - size(σ_aux)[1] + 1))
    σ_proposed = [σ_aux; σ_zeros_aux]


    # log_likelik
    log_likelik_aux_1  = log_likelik_seg[1:(index_s-1)]
    log_likelik_aux_2 = log_likelik_star_proposed
    log_likelik_aux_3 = log_likelik_seg[(index_s+2:(n_CP_current+1))]
    log_likelik_aux = [log_likelik_aux_1; log_likelik_aux_2; log_likelik_aux_3]
    log_likelik_zeros_aux = zeros(Float64, (n_CP_max - size(log_likelik_aux)[1] + 1))
    log_likelik_seg_proposed = [log_likelik_aux; log_likelik_zeros_aux ]
  end

  MH_ratio = log_likelik_ratio + log_prior_ratio + log_proposal_ratio + log_jacobian
  epsilon = min(1, exp(MH_ratio))
  U = rand()


  # ---- # Accept / Reject

  if (U <= epsilon)
    n_CP_sample[t] = n_CP_current - 1
    s_sample[1:n_CP_sample[t], t] = s_proposed[2:(end-1)]
    β_sample[:, :, t] = β_proposed
    ω_sample[:, :, t] = ω_proposed
    n_freq_sample[:, t] = n_freq_proposed
    σ_sample[:, t] = σ_proposed
    log_likelik_sample[:, t] = log_likelik_seg_proposed
    accepted = true
  else
    n_CP_sample[t] = n_CP_current
    s_sample[1:n_CP_sample[t], t] = s_current[2:(end-1)]
    β_sample[:, :, t] = β_seg
    ω_sample[:, :, t] = ω_seg
    n_freq_sample[1:(n_CP_sample[t] + 1), t] = n_freq_seg
    σ_sample[1:(n_CP_sample[t] + 1), t] = σ_seg
    log_likelik_sample[1:(n_CP_sample[t] + 1), t] = log_likelik_seg
    accepted = false
  end


  if (accepted == true)

    if (index_s == 1)

      # Suppose index_s == 1
      index_seg_update = collect((index_s+2):(n_CP_current + 1))

      # --- Updating segments not involved in death move (w.r.t t-1)
      for j in index_seg_update

        a = floor(Int64, s_current[j])
        if (j == n_CP_current + 1)
          b = floor(Int64, n_obs)
        else
          b = floor(Int64, (s_current[j+1] - 1))
        end

        global y = data[a:b]
        global n = length(y)
        global σ_RW_ω = (1/(σ_RW_ω_hyper*n))


        m_current = n_freq_sample[j, t-1]
        σ_current = σ_sample[j, t-1]
        β_current = β_sample[j, 1:(2*m_current+2), t-1]
        ω_current = ω_sample[j, 1:(m_current), t-1]

        MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
        n_freq_sample[j-1, t] = MCMC_S["m"]
        σ_sample[j-1, t] = MCMC_S["σ"]
        β_sample[j-1, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
        ω_sample[j-1, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
        log_likelik_sample[j-1, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

      end

      # --- Updating segment involved in death move (w.r.t )
      a = floor(Int64, s_current[index_s])
      b = floor(Int64, (s_current[index_s+1] - 1))
      global y = data[a:b]
      global n = length(y)
      global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

      m_current = n_freq_sample[index_s, t]
      σ_current = σ_sample[index_s, t]
      β_current = β_sample[index_s, 1:(2*m_current+2), t]
      ω_current = ω_sample[index_s, 1:(m_current), t]

      MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
      n_freq_sample[index_s, t] = MCMC_S["m"]
      σ_sample[index_s, t] = MCMC_S["σ"]
      β_sample[index_s, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
      ω_sample[index_s, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
      log_likelik_sample[index_s, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

    elseif (index_s == n_CP_current)

      index_seg_update = collect(1:(n_CP_current-1))

      # --- Updating segments not involved in death move (w.r.t t-1)
      for j in index_seg_update

        a = floor(Int64, s_current[j])
        if (j == n_CP_current + 1)
          b = floor(Int64, n_obs)
        else
          b = floor(Int64, (s_current[j+1] - 1))
        end
        global y = data[a:b]
        global n = length(y)
        global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

        m_current = n_freq_sample[j, t-1]
        σ_current = σ_sample[j, t-1]
        β_current = β_sample[j, 1:(2*m_current+2), t-1]
        ω_current = ω_sample[j, 1:(m_current), t-1]

        MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
        n_freq_sample[j, t] = MCMC_S["m"]
        σ_sample[j, t] = MCMC_S["σ"]
        β_sample[j, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
        ω_sample[j, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
        log_likelik_sample[j, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

      end

      # --- Updating segment involved in death move (w.r.t )
      a = floor(Int64, s_current[index_s])
      b = n_obs
      global y = data[a:b]
      global n = length(y)
      global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

      m_current = n_freq_sample[index_s, t]
      σ_current = σ_sample[index_s, t]
      β_current = β_sample[index_s, 1:(2*m_current+2), t]
      ω_current = ω_sample[index_s, 1:(m_current), t]

      MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
      n_freq_sample[index_s, t] = MCMC_S["m"]
      σ_sample[index_s, t] = MCMC_S["σ"]
      β_sample[index_s, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
      ω_sample[index_s, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
      log_likelik_sample[index_s, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

    else
      # --- Updating segments not involved in death move (w.r.t t-1)
      index_seg_update = vcat(collect(1:(index_s-1)), collect((index_s+2):(n_CP_current + 1)))

      for j in index_seg_update

        a = floor(Int64, s_current[j])
        if (j == n_CP_current + 1)
          b = floor(Int64, n_obs)
        else
          b = floor(Int64, (s_current[j+1] - 1))
        end
        global y = data[a:b]
        global n = length(y)
        global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

        m_current = n_freq_sample[j, t-1]
        σ_current = σ_sample[j, t-1]
        β_current = β_sample[j, 1:(2*m_current+2), t-1]
        ω_current = ω_sample[j, 1:(m_current), t-1]

        MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)


        if (j < index_s)

          n_freq_sample[j, t] = MCMC_S["m"]
          σ_sample[j, t] = MCMC_S["σ"]
          β_sample[j, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
          ω_sample[j, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
          log_likelik_sample[j, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])


        else

          n_freq_sample[j-1, t] = MCMC_S["m"]
          σ_sample[j-1, t] = MCMC_S["σ"]
          β_sample[j-1, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
          ω_sample[j-1, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
          log_likelik_sample[j-1, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

        end

      end

      # --- Updating segment involved in death move (w.r.t  t)
      a = floor(Int64, s_current[index_s])
      b = floor(Int64, s_current[index_s+2]) - 1
      global y = data[a:b]
      global n = length(y)
      global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

      m_current = n_freq_sample[index_s, t]
      σ_current = σ_sample[index_s, t]
      β_current = β_sample[index_s, 1:(2*m_current+2), t]
      ω_current = ω_sample[index_s, 1:(m_current), t]

      MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
      n_freq_sample[index_s, t] = MCMC_S["m"]
      σ_sample[index_s, t] = MCMC_S["σ"]
      β_sample[index_s, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
      ω_sample[index_s, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
      log_likelik_sample[index_s, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

    end
  else
    s_aux = vcat(1, s_sample[1:n_CP_current, t], n_obs)
    for j in 1:(n_CP_current + 1)
      a = floor(Int64, s_aux[j])
      if (j == n_CP_current + 1)
        b = floor(Int64, n_obs)
      else
        b = floor(Int64, (s_aux[j+1] - 1))
      end
      global y = data[a:b]
      global n = length(y)
      global σ_RW_ω = (1/(σ_RW_ω_hyper*n))

      m_current = n_freq_sample[j, t-1]
      σ_current = σ_sample[j, t-1]
      β_current = β_sample[j, 1:(2*m_current+2), t-1]
      ω_current = ω_sample[j, 1:(m_current), t-1]

      MCMC_S = RJMCMC_stationary_model(m_current, β_current, ω_current, σ_current, a, b, λ_S, c_S, ϕ_ω, ψ_ω, σ_β, ν0, γ0, n_freq_max)
      n_freq_sample[j, t] = MCMC_S["m"]
      σ_sample[j, t] = MCMC_S["σ"]
      β_sample[j, :, t] = vcat(MCMC_S["β"], zeros((2*n_freq_max+2)  - length(MCMC_S["β"])))
      ω_sample[j, :, t] = vcat(MCMC_S["ω"], zeros(n_freq_max - length(MCMC_S["ω"])))
      log_likelik_sample[j, t] = log_likelihood_segment_aux(y, MCMC_S["β"], MCMC_S["ω"], a, b, MCMC_S["σ"])

    end
  end

  output = Dict("n_CP" => n_CP_sample, "β" => β_sample, "σ" => σ_sample,
              "ω" => ω_sample, "s" => s_sample, "n_freq" => n_freq_sample,
              "log_likelik" => log_likelik_sample)

  return output
end
