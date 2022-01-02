abstract type ProcessModel <: RegressionModel end

#=
ProcessParams are the parameters of the parametric process model.
They determine the mean and covariance structure for a collection of observations
made at a given set of time points.
=#
abstract type ProcessParams end

#=
GaussianParams are the parameters of a Gaussian parametric process
model.
=#
mutable struct GaussianParams{T} <: ProcessParams where {T<:AbstractFloat}

    mean::Vector{T}

    scale::Vector{T}

    smooth::Vector{T}

    unexplained::Vector{T}
end

function GaussianParams()
    return GaussianParams(zeros(0), zeros(0), zeros(0), zeros(0))
end

abstract type AbstractXmat end

#=
The design matrices for a process model.
=#
mutable struct Xmat{T} <: AbstractXmat where {T<:Real}

    # Explanatory variables for the mean
    mean::Matrix{T}

    # Explanatory variables for the covariance scale, which uses log link
    scale::Matrix{T}

    # Explanatory variables for the covariance smoothness, which uses log link
    smooth::Matrix{T}

    # Explanatory variables for the unexplained standard deviation, which uses log link
    unexplained::Matrix{T}

end

#=
The L2 penalty parameters for fitting a process model.
=#
mutable struct Penalty

    mean::Float64

    scale::Float64

    smooth::Float64

    unexplained::Float64
end

#=
A specification of a process regression model to be fit using
maximum likelihood or L2 penalized maximum likelihood.
=#
mutable struct ProcessMLEModel{T} <: ProcessModel where {T<:Real}

    # Responses
    y::Vector{T}

    # The design matrices
    X::Xmat{T}

    # Times at which responses are observed
    time::Vector{T}

    # Beginning and ending indices (inclusive) for each group
    grp::Matrix{Int}

    # Group labels
    grplab::AbstractVector

    # Estimated parameters
    params::ProcessParams

    # Covariance matrix of parameters
    params_cov::Matrix{T}

    # If true, fix the unexplained variance terms at their starting values
    fix_unexplained::Vector{T}

    # L2 penalty parameters
    penalty::Penalty

end

function ProcessMLEModel(
    y::AbstractVector,
    X::Xmat,
    time::AbstractVector,
    grp::AbstractVector;
    fix_unexplained::AbstractVector = zeros(0),
    penalty = Penalty(0, 0, 0, 0),
)
    gp, _ = groupix(grp)
    return ProcessMLEModel(
        y,
        X,
        time,
        gp,
        grp,
        GaussianParams(),
        zeros(0, 0),
        fix_unexplained,
        penalty,
    )
end

#=
ProcessCovParams determine the covariance matrix for
a specific block (individual) and are obtained by
combining the ProcessParams with the covariates.
=#
abstract type ProcessCovPar end

#=
The covariance parameters for a Gaussian parametric process
regression model.
=#
mutable struct GaussianCovPar{T} <: ProcessCovPar where {T<:AbstractFloat}

    scale::Vector{T}
    lscale::Vector{T}

    smooth::Vector{T}
    lsmooth::Vector{T}

    unexplained::Vector{T}
    lunexplained::Vector{T}

end
