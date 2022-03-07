#=
This module implements maximum likelihood-based estimation (MLE) of
regression models for finite-dimensional observations made on
infinite-dimensional processes.  Currently only Gaussian models with
squared quadratic covariance structures are implemented.

The `ProcessMLEModel` class supports regression analyses on grouped
data, where the observations within a group are dependent (they are
made on the same realization of the Gaussian process).  Observations
in different groups are independent.

The covariance model has three types of parameters: scale parameters,
smoothing parameters, and unexplained variance parameters.  These
parameters are non-negative and are linked to covariates using a log
link.  The smoothing parameters mainly determine the autocorrelation,
the scale parameters mainly determine the process variance, and the
unexplained variance parameters provide an additional type of variance
that is not part of the process, and that is independent across all
observations within and between subjects.

The mean structure is specified as a linear model.  The covariance
parameters depend on covariates via a link function.
=#

using LinearAlgebra, Optim, StatsBase, FiniteDifferences, Distributions, Statistics

include("defs.jl")

# Return a 2 x m array, each column of which contains the indices
# spanning one group; also return the size of the largest group.
function groupix(g::AbstractVector)

    if !issorted(g)
        error("Group vector is not sorted")
    end

    ii = Int[]
    b, mx = 1, 0
    for i = 2:length(g)
        if g[i] != g[i-1]
            push!(ii, b, i - 1)
            mx = i - b > mx ? i - b : mx
            b = i
        end
    end
    push!(ii, b, length(g))
    mx = length(g) - b + 1 > mx ? length(g) - b + 1 : mx
    ii = reshape(ii, 2, div(length(ii), 2))

    return tuple(ii, mx)
end

# Evaluate the log-likelihood of the model at the given parameters.
function loglike(m::ProcessMLEModel{T}, par::GaussianParams{T})::T where {T<:Real}

    # Residuals
    resid = m.y - m.X.mean * par.mean

    # Get the log-likelihood
    ll = 0.0

    # Loop over the groups
    for (i1, i2) in eachcol(m.grp)

        cpar = covpar(m, par, i1, i2)

        # Get the explained covariance matrix for this person.
        cm = covmat(cpar, m.time[i1:i2])

        # Out of bounds
        if any(.!isfinite.(cm))
            return -Inf
        end

        # Update the likelihood
        re = resid[i1:i2]
        a, _ = eigen(cm)
        if minimum(a) <= 0
            return -Inf
        end
        ll -= 0.5 * sum(log, a)
        ll -= 0.5 * dot(re, cm \ re)

    end

    if size(m.penalty.mean, 1) > 0
        ll -= par.mean' * m.penalty.mean * par.mean
    end
    if size(m.penalty.scale, 1) > 0
        ll -= par.scale' * m.penalty.scale * par.scale
    end
    if size(m.penalty.smooth, 1) > 0
        ll -= par.smooth' * m.penalty.smooth * par.smooth
    end
    if size(m.penalty.unexplained, 1) > 0 && length(m.fix_unexplained) == 0
        ll -= par.unexplained' * m.penalty.unexplained * par.unexplained
    end

    return ll
end

# Evaluate the score (gradient of the log-likelihood).
function score(m::ProcessMLEModel{T}, par::GaussianParams{T}) where {T<:AbstractFloat}

    # Residuals
    resid = m.y - m.X.mean * par.mean

    score_mn = zeros(length(par.mean))
    score_sc = zeros(length(par.scale))
    score_sm = zeros(length(par.smooth))
    score_ux = zeros(length(par.unexplained))

    # Get the score contribution for each group
    for (i1, i2) in eachcol(m.grp)

        cpar = covpar(m, par, i1, i2)

        # Get the explained covariance matrix for this person.
        cm = covmat(cpar, m.time[i1:i2])
        if any(.!isfinite.(cm))
            return 0 .* score_mn, 0 .* score_sc, 0 .* score_sm, 0 .* score_ux
        end
        cmi = pinv(cm)

        jsc, jsm = jac(cpar, m.time[i1:i2])

        # The derivatives for the mean parameters.
        score_mn .+= m.X.mean[i1:i2, :]' * (cmi * resid[i1:i2])

        # The derivatives for the scaling parameters.
        rx = resid[i1:i2] * resid[i1:i2]'
        qm = 0.5 .* (cmi * rx * cmi)
        scx = cpar.scale .* m.X.scale[i1:i2, :]
        for i in eachindex(jsc)
            c1 = 2 * sum(qm[i, :] .* jsc[i]) - qm[i, i] * jsc[i][i]
            c2 = 2 * sum(cmi[i, :] .* jsc[i]) - cmi[i, i] * jsc[i][i]
            score_sc .+= (c1 - c2 / 2) * scx[i, :]
        end

        # The derivatives for the smoothness parameters.
        smx = cpar.smooth .* m.X.smooth[i1:i2, :]
        for i in eachindex(jsm)
            c1 = 2 * sum(qm[i, :] .* jsm[i]) - qm[i, i] * jsm[i][i]
            c2 = 2 * sum(cmi[i, :] .* jsm[i]) - cmi[i, i] * jsm[i][i]
            score_sm += (c1 - c2 / 2) * smx[i, :]
        end

        # The derivatives with respect to the unexplained standard
        # deviation parameters
        if length(m.fix_unexplained) == 0
            sux = (cpar.unexplained .^ 2) .* m.X.unexplained[i1:i2, :]
            score_ux .-= sux' * diag(cmi)
            bm = cmi * rx * cmi
            score_ux .+= sux' * diag(bm)
        end
    end

    if size(m.penalty.mean, 1) > 0
        score_mn .-= 2 * m.penalty.mean * par.mean
    end
    if size(m.penalty.scale, 1) > 0
        score_sc .-= 2 * m.penalty.scale * par.scale
    end
    if size(m.penalty.smooth, 1) > 0
        score_sm .-= 2 * m.penalty.smooth * par.smooth
    end
    if size(m.penalty.unexplained, 1) > 0 && length(m.fix_unexplained) == 0
        score_ux .-= 2 * m.penalty.unexplained * par.unexplained
    end

    return score_mn, score_sc, score_sm, score_ux
end

function unpack(m::ProcessMLEModel, x::Vector{T}) where {T<:Real}
    pmn, psc = size(m.X.mean, 2), size(m.X.scale, 2)
    psm, pux = size(m.X.smooth, 2), size(m.X.unexplained, 2)
    return GaussianParams(
        x[1:pmn],
        x[pmn+1:pmn+psc],
        x[pmn+psc+1:pmn+psc+psm],
        x[pmn+psc+psm+1:end],
    )
end

function pack(par::ProcessParams)
    return vcat(par.mean, par.scale, par.smooth, par.unexplained)
end

function getstart(m::ProcessMLEModel)

    # OLS for the mean parameters
    u, s, v = svd(m.X.mean)
    pmean = v * diagm(1 ./ s) * u' * m.y
    fit = m.X.mean * pmean

    resid = m.y - fit
    sd = std(resid)

    pscale = zeros(size(m.X.scale, 2))
    pscale[1] = log(sd)
    psmooth = zeros(size(m.X.smooth, 2))
    punexplained =
        length(m.fix_unexplained) > 0 ? m.fix_unexplained : zeros(size(m.X.unexplained, 2))
    if length(m.fix_unexplained) == 0
        punexplained[1] = log(sd)
    end

    return vcat(pmean, pscale, psmooth, punexplained)
end

function revert_standardize(m::ProcessMLEModel, skip_se::Bool)
    # Revert the standardization in the parameters
    m.params.mean .*= m.ymom[2]
    m.params.mean[1] += m.ymom[1]
    m.params.scale[1] += log(m.ymom[2])
    if length(m.fix_unexplained) == 0
        m.params.unexplained[1] += log(m.ymom[2])
    end

    # Revert the standardization in the standard errors
    if !skip_se
        pmn = size(m.X.mean, 2)
        m.params_cov[1:pmn, 1:pmn] .*= m.ymom[2]^2
    end

    # Return the outcome to its original scale
    m.y = m.y * m.ymom[2] .+ m.ymom[1]
end

function _fit!(
    m::ProcessMLEModel,
    verbose::Bool,
    maxiter::Int,
    atol::Float64,
    rtol::Float64,
    start;
    maxiter_gd = 20,
    algorithm = LBFGS(),
    g_tol = 1e-8,
    skip_se = false,
)

    pmn, psc = size(m.X.mean, 2), size(m.X.scale, 2)
    psm, pux = size(m.X.smooth, 2), size(m.X.unexplained, 2)

    # Wrap the log-likelihood and score functions
    # for minimization.
    f = function (x::Vector{T}) where {T<:Real}
        return -loglike(m, unpack(m, x))
    end

    g! = function (g, x)
        score_mn, score_sc, score_sm, score_ux = score(m, unpack(m, x))
        g .= vcat(score_mn, score_sc, score_sm, score_ux)
        g .*= -1
    end

    if isnothing(start)
        start = getstart(m)
    end

    # Refine starting values using gradient sescent
    r = optimize(
        f,
        g!,
        typeof(start) <: ProcessParams ? pack(start) : start,
        GradientDescent(),
        Optim.Options(iterations = maxiter_gd, show_trace = verbose),
    )

    r = optimize(
        f,
        g!,
        Optim.minimizer(r),
        algorithm,
        Optim.Options(iterations = maxiter, show_trace = verbose, g_tol = g_tol),
    )

    if !Optim.converged(r)
        println("ProcessRegression fitting did not converge")
    end

    b = Optim.minimizer(r)
    m.params = unpack(m, b)

    # Use numerical differentiation to get the Hessian.
    if !skip_se
        score1 = function (x)
            g = zeros(length(x))
            score_mn, score_sc, score_sm, score_ux = score(m, unpack(m, x))
            return vcat(score_mn, score_sc, score_sm, score_ux)
        end
        hess = jacobian(central_fdm(12, 1), score1, b)[1]
        hess = Symmetric(-(hess + hess') ./ 2)
        if length(m.fix_unexplained) > 0
            q = size(hess, 1) - length(m.fix_unexplained)
            hess = hess[1:q, 1:q]
        end
        m.params_cov = inv(hess)
    end

    if length(m.ymom) == 2
        revert_standardize(m, skip_se)
    end
end

function StatsBase.fit!(
    m::ProcessModel;
    verbose::Bool = false,
    maxiter::Integer = 100,
    atol::Real = 1e-6,
    rtol::Real = 1e-6,
    start = nothing,
    kwargs...,
)
    _fit!(m, verbose, maxiter, atol, rtol, start; kwargs...)
end

function fit(
    ::Type{M},
    X::Xmat{T},
    y::Vector{T},
    ti::Vector{T},
    grp::AbstractVector;
    dofit::Bool = true,
    penalty = nothing,
    fitargs...,
) where {M<:ProcessModel,T<:Real}

    if !(
        size(X.mean, 1) ==
        size(X.scale, 1) ==
        size(X.smooth, 1) ==
        size(y, 1) ==
        size(grp, 1)
    )
        throw(DimensionMismatch("Number of rows in X matrices, y and g must match"))
    end

    if !(size(X.unexplained, 1) in [0, length(y)])
        throw(
            DimensionMismatch(
                "X.unexplained must be either empty, or have the same number of rows as the other X matrices.",
            ),
        )
    end

    c = ProcessMLEModel(y, X, ti, grp, penalty)

    return dofit ? fit!(c; fitargs...) : c
end

function vcov(m::ProcessModel)
    if size(m.params_cov, 1) > 0
        return m.params_cov
    else
        # The model was fit with skip_se == true
        p = length(coef(m))
        return zeros(p, p)
    end
end

function getnames(m::ProcessModel)::Vector{String}
    names = String[]
    for i = 1:size(m.X.mean, 2)
        push!(names, "mean_$(i)")
    end
    for i = 1:size(m.X.scale, 2)
        push!(names, "scale_$(i)")
    end
    for i = 1:size(m.X.smooth, 2)
        push!(names, "smooth_$(i)")
    end
    for i = 1:size(m.X.unexplained, 2)
        push!(names, "unexplained_$(i)")
    end

    return names
end

coef(m::ProcessModel) = pack(m.params)

function coeftable(m::ProcessModel; level::Real = 0.95)

    cc = pack(m.params)
    va = diag(vcov(m))
    if minimum(va) < 0
        println("Warning: the Hessian matrix is not positive definite")
    end
    if length(m.fix_unexplained) > 0
        va = vcat(va, zeros(length(m.fix_unexplained)))
    end
    va[va.<0] .= Inf
    se = sqrt.(va)
    zz = cc ./ se
    p = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se * quantile(Normal(), (1 - level) / 2)
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    return CoefTable(
        hcat(cc, se, zz, p, cc + ci, cc - ci),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"],
        getnames(m),
        4,
        3,
    )
end

# Map the structural model parameters to the specific scale, smooth and unexplained
# variance parameters for a particular group.
function covpar(m::ProcessMLEModel, par::GaussianParams, i1::Int, i2::Int)::GaussianCovPar

    lsc = m.X.scale[i1:i2, :] * par.scale
    lsm = m.X.smooth[i1:i2, :] * par.smooth
    lux = m.X.unexplained[i1:i2, :] * par.unexplained

    gcp = GaussianCovPar(lsc, lsm, lux)
    return gcp
end

#=
Following Paciorek et al, the covariance between observations with
index `i` and `j` is given by:

  s[i] \cdot s[j] \cdot h(|time[i] - time[j]| / \sqrt{(u[i] + u[j]) /
  2}) \cdot \frac{u[i]^{1/4}u[j]^{1/4}}{\sqrt{(u[i] + u[j])/2}}
=#
function covmat(c::GaussianCovPar{T}, time::Vector{T})::Matrix{T} where {T<:AbstractFloat}

    sc, lsc, sm, lsm = c.scale, c.lscale, c.smooth, c.lsmooth
    ux, lux = c.unexplained, c.lunexplained

    @assert length(time) == length(sc) == length(sm)
    @assert length(ux) in [0, length(time)]

    m = length(time)
    cm = zeros(m, m)
    for i = 1:m
        for j = 1:m
            dt = time[i] - time[j]
            sa = (sm[i] + sm[j]) / 2

            # Minimum and maximum of the log smoothing parameters
            lsm1, lsm2 = if lsm[i] < lsm[j]
                lsm[i], lsm[j]
            else
                lsm[j], lsm[i]
            end
            lsa = log1p(exp(lsm1 - lsm2)) - log(2)

            sca = lsc[i] + lsc[j]
            sma = (lsm1 - lsm2) / 4
            q = dt^2 / sa
            cm[i, j] = exp(-q / 2 + sca + sma - lsa / 2)
        end
    end

    for i = 1:m
        cm[i, i] += ux[i]^2
    end

    return cm
end

function jac(c::GaussianCovPar{T}, time::Vector{T}) where {T<:AbstractFloat}

    sc, sm, ux = c.scale, c.smooth, c.unexplained
    @assert length(time) == length(sc) == length(sm)

    p = length(time)
    dt = time .- time'
    sa = (sm .+ sm') ./ 2
    sds = sqrt.(sa)
    dtt = dt .^ 2
    qmat = dtt ./ sa
    eqm = exp.(-qmat ./ 2)
    eqmx = exp.(-qmat ./ 2 + 2 * log.(abs.(dt)) - log.(sa))
    sm4 = (sm * sm') .^ 0.25
    cmx = eqm .* sm4 ./ sds

    # Derivatives with respect to the smoothing parameters.
    jsm = Vector{T}[]
    for i in eachindex(sm)
        dbottom = 0.25 ./ sds[:, i]
        dbottom[i] *= 2
        dtop = 0.5 .* eqmx[:, i] ./ (sm .+ sm[i])
        dtop[i] *= 2
        b = dtop ./ sds[:, i] .- 2 .* eqm[:, i] .* dbottom ./ (sm .+ sm[i])
        c = eqm[:, i] ./ sds[:, i]
        fi = 0.25 .* sm .^ 0.25 ./ sm[i] .^ 0.75
        fi[i] = 0.5 ./ sm[i] .^ 0.5
        b = c .* fi + b .* sm4[:, i]
        b .*= sc[i] .* sc
        push!(jsm, b)
    end

    # Derivatives with respect to the scaling parameters.
    jsc = Vector{T}[]
    for i in eachindex(sc)
        b = cmx[i, :] .* sc
        b[i] *= 2
        push!(jsc, b)
    end

    return jsc, jsm
end
