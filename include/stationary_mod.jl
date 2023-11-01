function eye(n) #MAX: NEEDED TO ADD THIS
    return 1.0*Matrix(I,n,n)
end


# Function: get design matrix with Fourier basis function.
function get_X(ω, a, b)

  M = length(ω)
  time = a:b
  X = ones(length(time))
  X = hcat(X, time)

  for j in 1:M
    X = hcat(X, cos.(2π*time*ω[j]),sin.(2π*time*ω[j]))
  end
  return X
end

# Function: get detrended data via linear regression over time
#           (and trend)
function detrend(data)

  N = length(data)
  time = 1:N
  covariate_time = ones(N, 2)
  covariate_time[:, 2] = time
  b = inv(covariate_time'*covariate_time)*covariate_time'*data
  trend = covariate_time * b
  y = data - trend    # De-trended data

  return(y, trend)
end

# Function: log likelihood for β and ω
function log_likelik(β, ω, a, b)
  X = get_X(ω, a, b)
  out = -(n*log(2π))/2 -(n*log(σ^2))/2 - sum((y - X*β).^2)/(2*(σ^2))
  return out
end

# Function: log_posterior_β
function log_posterior_β_stationary(β, ω, σ, σ_β, a, b)
  X = get_X(ω, a, b)
  f = (-sum((y - X*β).^2)/(2*(σ^2)) - ((β'*β)/(2*(σ_β^2))))[1]
  return f
end

# Function: log_posterior_ω
function log_posterior_ω_stationary(ω, β, a, b)
  X = get_X(ω, a, b)
  f = (-sum((y - X*β).^2)/(2*(σ^2)))[1]
  return f
end

# Function: sample uniformly from continuous disjoint subintervals
function sample_uniform_continuous_intervals(n_sample::Int64, intervals)

  out = zeros(n_sample)
  n_intervals = length(intervals)

  # Getting length of each interval
  len_intervals = zeros(n_intervals)
  for k in 1:n_intervals
      aux  = intervals[k][2]-intervals[k][1]
      if (aux < 0) aux = 0 end
      len_intervals[k] = aux
  end

  # Getting proportion of each interval
  weights = zeros(n_intervals)
  for k in 1:n_intervals
    weights[k] = len_intervals[k]/sum(len_intervals)
  end

  # Getting samples
  for j in 1:n_sample
    indicator = wsample(1:n_intervals, weights)
    out[j] = rand(Uniform(intervals[indicator][1], intervals[indicator][2]))
  end

  return out
end


# Function: segment_model, within step.
function within_move_stationary(m_current, β_current, ω_current,
                                      σ_current, a, b, λ, c, ϕ_ω, ψ_ω)


  if (2*length(ω_current) != (length(β_current) - 2)) error("dimension mismatch, ω and β") end


  global σ = σ_current

  # -------------------------- Sampling frequencies -------------------------

  period = periodogram(y)
  p = period.power
  p_norm = p ./sum(p)
  freq = period.freq

  U = rand()

  # ------------------ Gibbs step (FFT) ----------------
  if (U <= δ_ω_mixing)

    ω_current_aux = copy(ω_current)

    for j in 1:(m_current)
      ω_curr = copy(ω_current_aux)

      # Avoiding a vector with two same frequencies (D column would be linear dependent)
      aux_temp = false

      while (aux_temp == false)

        # Proposing frequencies
        global ω_star = sample(freq, Weights(p_norm))
        global ω_prop = copy(ω_curr)

        # Updating j-th component
        ω_prop[j] = ω_star

        if (! (any(vcat(ω_prop[1:(j-1)], ω_prop[(j+1):end]) .== ω_star)))
          aux_temp = true
        end

      end

      log_likelik_ratio = log_posterior_ω_stationary(ω_prop, β_current, a, b) -
                          log_posterior_ω_stationary(ω_curr, β_current, a, b)

      log_proposal_ratio = log(p_norm[searchsortedlast(freq, ω_curr[j])]) -
                           log(p_norm[searchsortedlast(freq, ω_star)])

      MH_ratio = exp(log_likelik_ratio + log_proposal_ratio)[1]

      U = rand()

      if (U <= min(1, MH_ratio))
        ω_current_aux = ω_prop
      else
        ω_current_aux = ω_curr
      end
    end

    ω_out = sort(ω_current_aux)

  # ------------------ Random Walk MH ------------------
  else

    ω_current_aux = copy(ω_current)

    for j in 1:m_current

      ω_curr = copy(ω_current_aux)
      aux_temp = false

      # Proposed frequency has to lie within [0,  0.5]
      while (aux_temp == false)
        global ω_star = rand(Normal(ω_current[j], σ_RW_ω), 1)[1]
        if !(ω_star <= 0 || ω_star >= 0.5)
          aux_temp = true
        end
      end

      global ω_prop = copy(ω_curr)

      # Updating the j-th component
      ω_prop[j] = ω_star

      log_likelik_ratio = log_posterior_ω_stationary(ω_prop, β_current, a, b) -
                          log_posterior_ω_stationary(ω_curr, β_current, a, b)

      MH_ratio = exp.(log_likelik_ratio)[1]

      U = rand()

      if (U <= min(1, MH_ratio))
        ω_current_aux = ω_prop
      else
        ω_current_aux = ω_curr
      end
    end

    ω_out = sort(ω_current_aux)

  end




  # ---------- Sampling β

  X_post = get_X(ω_out, a, b)

  β_var_post = inv(eye(2*m_current+2)/(σ_β^2) + (X_post'*X_post)/(σ^2))
  β_var_post = (β_var_post' + β_var_post)/2
  β_mean_post = β_var_post*((X_post'*y)/(σ^2))

β_var_post=Hermitian(β_var_post)            #MAX: NEEDED TO FIX THIS
  β_out = rand(MultivariateNormal(β_mean_post, β_var_post), 1)


  # ------- Sampling σ

  #X_post = get_X(ω_out, a, b)
  res_var = sum((y - X_post*β_out).^2)

  ν_post = (n + ν0)/2
  γ_post = (γ0 + res_var)/2

  σ_out  = sqrt.(rand(InverseGamma(ν_post, γ_post), 1))[1]



  output = Dict("β" => β_out,
                "ω" => ω_out,
                "σ" => σ_out)
end

# Function: segment_model, birth step.
function birth_move_stationary(m_current, β_current, ω_current,
                               σ_current, a, b, λ, c, ϕ_ω, ψ_ω)

  if (2*length(ω_current) != (length(β_current) - 2)) error("dimension mismatch, ω and β") end

  global σ = σ_current

  m_proposed = m_current + 1


  # - Proposing ω
  ω_current_aux = sort(vcat(0, ω_current, ϕ_ω))
  support_ω = Array{Vector{Float64}}(undef,m_current + 1)
  for k in 1:(m_current + 1)
    support_ω[k] = [ω_current_aux[k] + ψ_ω, ω_current_aux[k+1] - ψ_ω]
  end
  length_support_ω = ϕ_ω - (2*(m_current + 1)*ψ_ω)

  ω_star = sample_uniform_continuous_intervals(1, support_ω)[1]
  ω_proposed = sort(vcat(ω_current, ω_star))


  # - Proposing β ∼ Normal(β̂_prop, Σ̂_prop)

  X_prop = get_X(ω_proposed, a, b)
  β_var_prop = inv(eye(2*m_proposed+2)/(σ_β^2) + (X_prop'*X_prop)/(σ^2))
  β_mean_prop = β_var_prop*((X_prop'*y)/(σ^2))
  β_proposed = rand(MvNormal(β_mean_prop, 0.5*(β_var_prop + β_var_prop')), 1)


  # - Obtaining β̂_curr, Σ̂_curr (for proposal ratio)
  X_curr = get_X(ω_current, a, b)
  β_var_curr = inv(eye(2*m_current+2)/(σ_β^2) + (X_curr'*X_curr )/(σ^2))
  β_mean_curr = β_var_curr*((X_curr'*y)/(σ^2))


  # --- Proposing σ

  X_post = get_X(ω_proposed, a, b)
  res_var = sum((y - X_post*β_proposed).^2)
  ν_post = (n + ν0)/2
  γ_post = (γ0 + res_var)/2

  σ_proposed = sqrt.(rand(InverseGamma(ν_post, γ_post), 1))[1]

  # ----- Evaluating acceptance probability

  # --- Log likelihood ratio
  log_likelik_prop = log_likelik(β_proposed, ω_proposed, a , b)
  log_likelik_curr = log_likelik(β_current, ω_current, a, b)
  log_likelik_ratio = log_likelik_prop - log_likelik_curr


  # --- Log prior ratio
  log_m_prior_ratio = log.(pdf(Poisson(λ), m_proposed)) - log(pdf(Poisson(λ), m_current))
  log_β_prior_ratio = (log.(pdf(MvNormal(zeros(2*m_proposed+2), (σ_β^2)*eye(2*m_proposed+2)) , β_proposed))[1]) -
                       log.(pdf(MvNormal(zeros(2*m_current+2), (σ_β^2)*eye(2*m_current+2)), β_current))
  log_ω_prior_ratio = log(2)
  log_σ2_prior_ratio = log(pdf(InverseGamma(ν0/2, γ0/2), σ_proposed^2)) -
                       log(pdf(InverseGamma(ν0/2, γ0/2), σ_current^2))


  log_prior_ratio = log_m_prior_ratio + log_β_prior_ratio +
                    log_ω_prior_ratio + log_σ2_prior_ratio

  # Preparing object for proposal ratio σ2

  X_post_current = get_X(ω_current, a, b)
  X_post_proposed = get_X(ω_proposed, a, b)
  res_var_current = sum((y - X_post_current*β_current).^2)
  res_var_proposed = sum((y - X_post_proposed*β_proposed).^2)

  ν_post = (n + ν0)/2
  γ_post_current = (γ0 + res_var_current)/2
  γ_post_proposed  = (γ0 + res_var_proposed)/2


  if (pdf(InverseGamma(ν_post, γ_post_current), σ_current^2) == 0.0)
    log_proposal_σ2_current = 0.0
  else
    log_proposal_σ2_current = log(pdf(InverseGamma(ν_post, γ_post_current), σ_current^2))
  end

  if (pdf(InverseGamma(ν_post, γ_post_proposed), σ_proposed^2) == 0.0)
    log_proposal_σ2_proposed = 0.0
  else
    log_proposal_σ2_proposed = log(pdf(InverseGamma(ν_post, γ_post_proposed), σ_proposed^2))
  end


  # --- Log proposal ratio
  log_proposal_β_prop = (-0.5*(β_proposed - β_mean_prop)'*inv(β_var_prop)*(β_proposed - β_mean_prop) .-
                            log(sqrt(det(2π*β_var_prop))))[1]
  log_proposal_β_current = (-0.5*(β_current - β_mean_curr)'*inv(β_var_curr)*(β_current - β_mean_curr) -
                            log(sqrt(det(2π*β_var_curr))))[1]
  log_proposal_ω_proposed = log((1/length_support_ω))
  log_proposal_ω_current = log(1/m_proposed)


  log_proposal_birth_move = log(c*min(1, pdf(Poisson(λ), m_proposed)/pdf(Poisson(λ), m_current)))
  log_proposal_death_move = log(c*min(1, pdf(Poisson(λ), m_current)/pdf(Poisson(λ), m_proposed)))
  log_proposal_ratio = log_proposal_death_move - log_proposal_birth_move +
                       log_proposal_ω_current - log_proposal_ω_proposed + log_proposal_β_current -
                       log_proposal_β_prop + log_proposal_σ2_current - log_proposal_σ2_proposed



  # --- MH acceptance step
  MH_ratio_birth = log_likelik_ratio + log_prior_ratio + log_proposal_ratio
  epsilon_birth = min(1, exp(MH_ratio_birth))

  U = rand()

  if (U <= epsilon_birth)
    β_out = β_proposed
    ω_out = ω_proposed
    σ_out = σ_proposed
    accepted = true
  else
    β_out = β_current
    ω_out = ω_current
    σ_out = σ_current
    accepted = false
  end

  output = Dict("β" => β_out, "ω" => ω_out, "σ" => σ_out,
                "accepted" => accepted,
                "ω_star" => ω_star)
end

# Function: segment_model, death step.
function death_move_stationary(m_current, β_current, ω_current,
                               σ_current, a, b, λ, c, ϕ_ω, ψ_ω)


  global σ = σ_current

  if (2*length(ω_current) != (length(β_current)-2 )) error("dimension mismatch, ω and β") end

  m_proposed = m_current - 1
  index = sample(1:m_current)
  ω_proposed = vcat(ω_current[1:(index-1)], ω_current[(index+1):end])


  # # - Proposing β ∼ Normal(β_max, Σ_max)
  # global ω = ω_proposed
  #
  # β_mean_prop = optimize(neg_f_posterior_β_stationary, neg_g_posterior_β_stationary!, neg_h_posterior_β_stationary!,
  #         zeros(2*m_proposed+2), BFGS()).minimizer
  # β_var_prop = inv(neg_hess_log_posterior_β_stationary(β_mean_prop, ω, σ, σ_β, a, b))
  # β_proposed = rand(MvNormal(β_mean_prop, 0.5*(β_var_prop + β_var_prop')), 1)
  #
  # global ω = ω_current
  # β_mean_curr = optimize(neg_f_posterior_β_stationary, neg_g_posterior_β_stationary!, neg_h_posterior_β_stationary!,
  #         zeros(2*m_current+2), BFGS()).minimizer
  # β_var_curr = inv(neg_hess_log_posterior_β_stationary(β_mean_curr, ω, σ, σ_β, a, b))



  # - Proposing β ∼ Normal(β̂_prop, Σ̂_prop)

  X_prop = get_X(ω_proposed, a, b)
  β_var_prop = inv(eye(2*m_proposed+2)/(σ_β^2) + (X_prop'*X_prop)/(σ^2))
  β_mean_prop = β_var_prop*((X_prop'*y)/(σ^2))
  β_proposed = rand(MvNormal(β_mean_prop, 0.5*(β_var_prop + β_var_prop')), 1)


  # - Obtaining β̂_curr, Σ̂_curr (for proposal ratio)
  X_curr = get_X(ω_current, a, b)
  β_var_curr = inv(eye(2*m_current+2)/(σ_β^2) + (X_curr'*X_curr )/(σ^2))
  β_mean_curr = β_var_curr*((X_curr'*y)/(σ^2))

  length_support_ω = ϕ_ω - (2*(m_current)*ψ_ω)


  # --- Proposing σ

  X_post = get_X(ω_proposed, a, b)
  res_var = sum((y - X_post*β_proposed).^2)
  ν_post = (n + ν0)/2
  γ_post = (γ0 + res_var)/2

  σ_proposed = sqrt.(rand(InverseGamma(ν_post, γ_post), 1))[1]

  # ----- Evaluating acceptance probability

  # --- Log likelihood ratio
  log_likelik_prop = log_likelik(β_proposed, ω_proposed, a, b)
  log_likelik_curr = log_likelik(β_current, ω_current, a, b)
  log_likelik_ratio = log_likelik_prop - log_likelik_curr

  # --- Log prior ratio
  log_m_prior_ratio = log(pdf(Poisson(λ), m_proposed)) - log(pdf(Poisson(λ), m_current))
  log_β_prior_ratio = (log.(pdf(MvNormal(zeros(2*m_proposed+2), (σ_β^2)*eye(2*m_proposed+2)) , β_proposed))[1]) -
                       log.(pdf(MvNormal(zeros(2*m_current+2), (σ_β^2)*eye(2*m_current+2)), β_current))
  log_ω_prior_ratio = log(0.5)
  log_σ2_prior_ratio = log(pdf(InverseGamma(ν0/2, γ0/2), σ_proposed^2)) -
                       log(pdf(InverseGamma(ν0/2, γ0/2), σ_current^2))
  log_prior_ratio = log_m_prior_ratio + log_β_prior_ratio +
                    log_ω_prior_ratio + log_σ2_prior_ratio


  # Preparing object for proposal ratio σ2

  X_post_current = get_X(ω_current, a, b)
  X_post_proposed = get_X(ω_proposed, a, b)
  res_var_current = sum((y - X_post_current*β_current).^2)
  res_var_proposed = sum((y - X_post_proposed*β_proposed).^2)

  ν_post = (n + ν0)/2
  γ_post_current = (γ0 + res_var_current)/2
  γ_post_proposed  = (γ0 + res_var_proposed)/2


  if (pdf(InverseGamma(ν_post, γ_post_current), σ_current^2) == 0.0)
    log_proposal_σ2_current = 0.0
  else
    log_proposal_σ2_current = log(pdf(InverseGamma(ν_post, γ_post_current), σ_current^2))
  end

  if (pdf(InverseGamma(ν_post, γ_post_proposed), σ_proposed^2) == 0.0)
    log_proposal_σ2_proposed = 0.0
  else
    log_proposal_σ2_proposed = log(pdf(InverseGamma(ν_post, γ_post_proposed), σ_proposed^2))
  end

  # --- Log proposal ratio
  log_proposal_β_prop = (-0.5*(β_proposed - β_mean_prop)'*inv(β_var_prop)*(β_proposed - β_mean_prop) .- #MAX: FIXED THIS
                            log(sqrt(det(2π*β_var_prop))))[1]
  log_proposal_β_current = (-0.5*(β_current - β_mean_curr)'*inv(β_var_curr)*(β_current - β_mean_curr) -
                            log(sqrt(det(2π*β_var_curr))))[1]
  log_proposal_ω_current = log((1/length_support_ω))
  log_proposal_ω_proposed = log(1/m_current)
  log_proposal_birth_move = log(c*min(1, pdf(Poisson(λ), m_current)/pdf(Poisson(λ), m_proposed)))
  log_proposal_death_move = log(c*min(1, pdf(Poisson(λ), m_proposed)/pdf(Poisson(λ), m_current)))

  log_proposal_ratio = log_proposal_birth_move - log_proposal_death_move +
                       log_proposal_ω_current - log_proposal_ω_proposed + log_proposal_β_current -
                       log_proposal_β_prop + log_proposal_σ2_current - log_proposal_σ2_proposed



  # --- MH acceptance step
  MH_ratio_death = log_likelik_ratio + log_prior_ratio + log_proposal_ratio
  epsilon_death = min(1, exp(MH_ratio_death))

  U = rand()
  if (U <= epsilon_death)
    β_out = β_proposed
    ω_out = ω_proposed
    σ_out = σ_proposed
    accepted = true
  else
    β_out = β_current
    ω_out = ω_current
    σ_out = σ_current
    accepted = false
  end

  output = Dict("β" => β_out, "ω" => ω_out, "σ" => σ_out,
                "accepted" => accepted)
end

# Function: RJMCMC for 'stationary' time series (i.e. no change-points)
# y is global, n is global
function RJMCMC_stationary_model(m_current, β_current, ω_current,
                                  σ_current, a, b, λ, c, ϕ_ω, ψ_ω, n_freq_max)

  if ( (length(β_current)-2) != (2*m_current) || (length(ω_current) != m_current) )
    error("dimension mismatch, ω and β") end

  # If m == 1, then either birth or within model move
  if (m_current == 1)

    birth_prob = c*min(1, (pdf(Poisson(λ), 2)/
                   pdf(Poisson(λ), 1)))
    U = rand()

    if (U <= birth_prob)
      MCMC = birth_move_stationary(m_current, β_current, ω_current, σ_current, a, b, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current + Int64(MCMC["accepted"])
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]

    else
      MCMC = within_move_stationary(m_current, β_current, ω_current, σ_current, a, b, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]
    end

  # If m == n_freq_max, then either death or within model move
  elseif (m_current == n_freq_max)

    death_prob = c*min(1, (pdf(Poisson(λ), n_freq_max - 1)/
                   pdf(Poisson(λ), n_freq_max)))
    U = rand()
    if (U <= death_prob)
      MCMC = death_move_stationary(m_current, β_current, ω_current, σ_current, a, b, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current - Int64(MCMC["accepted"])
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]
    else
      MCMC = within_move_stationary(m_current, β_current, ω_current, σ_current, a, b, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]
    end

  else

    birth_prob = c*min(1, (pdf(Poisson(λ), m_current + 1)/
                   pdf(Poisson(λ), m_current)))
    death_prob = c*min(1, (pdf(Poisson(λ), m_current - 1)/
                   pdf(Poisson(λ), m_current)))

    U = rand()

    # ----- Birth
    if (U <= birth_prob)
      MCMC = birth_move_stationary(m_current, β_current, ω_current, σ_current, a, b, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current + Int64(MCMC["accepted"])
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]

    # ----- Death
    elseif ((U > birth_prob) && (U <= (birth_prob + death_prob)))

      MCMC = death_move_stationary(m_current, β_current, ω_current, σ_current, a, b, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current - Int64(MCMC["accepted"])
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]

    # ---- Within model
    else
      MCMC = within_move_stationary(m_current, β_current, ω_current, σ_current, a, b, λ, c, ϕ_ω, ψ_ω)
      m_out = m_current
      β_out = MCMC["β"]
      ω_out = MCMC["ω"]
      σ_out = MCMC["σ"]
    end

  end

  return Dict("m" => m_out, "β" => β_out,
               "ω" => ω_out, "σ" => σ_out)
end
