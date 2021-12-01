module ProcessRegression

using StatsBase: CoefTable, StatisticalModel, RegressionModel
import StatsBase: coef, coeftable, vcov, stderr, fit

export GaussianParams, GaussianCovPar, ProcessParams, ProcessCovPar
export ProcessModel, ProcessMLEModel, Xmat
export fit, fit!, coef, coeftable, covmat, jac, covpar, loglike, score, emulate, groupix

include("gpr.jl")
include("emulate.jl")

end
