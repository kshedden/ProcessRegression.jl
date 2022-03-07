using ProcessRegression, Test, LinearAlgebra, StableRNGs, Statistics

function armat(ti)
    m = length(ti)
    cm = zeros(m, m)
    for i = 1:m
        for j = 1:m
            d = ti[i] - ti[j]
            cm[i, j] = exp(-d^2 / 2)
        end
        cm[i, i] += 1
    end
    return cm
end

function get_basis(nb, age, scale)
    xb = ones(length(age), nb)
    age1 = minimum(age)
    age2 = maximum(age)
    for (j, v) in enumerate(range(age1, age2, length = nb - 1))
        u = (age .- v) ./ scale
        v = exp.(-u .^ 2 ./ 2)
        xb[:, j+1] = v .- mean(v)
    end
    return xb
end

@testset "Check likelihood gradient using numerical derivatives" begin

    rng = StableRNG(123)

    par = GaussianParams([1.0, -1.5], [2.0, 0.0], [1.5, 0.0], [1.0, 0])

    n = 1000 # Number of groups
    m = 5    # Average number of observations per group

    ti = range(0, 1, length = m)
    tim = kron(ones(n), ti)
    x = zeros(n * m, 2)
    x[:, 1] .= 1
    x[:, 2] = tim
    X = Xmat(x, x, x, x)
    grp = kron(1:n, ones(m))
    pm = ProcessMLEModel(zeros(0), X, tim, grp; standardize = false)
    y = emulate(pm; par = par, rng = rng)
    pm = ProcessMLEModel(y, X, tim, grp; standardize = false)

    ll = loglike(pm, par)
    score_mn, score_sc, score_sm, score_ux = score(pm, par)

    # Check the score for mean parameters
    ee = 1e-7
    nscore_mn = zeros(length(score_mn))
    for j in eachindex(score_mn)
        par1 = deepcopy(par)
        par1.mean[j] += ee
        ll1 = loglike(pm, par1)
        nscore_mn[j] = (ll1 - ll) / ee
    end
    @test isapprox(nscore_mn, score_mn, rtol = 1e-4, atol = 1e-4)

    # Check the score for scale parameters
    ee = 1e-6
    nscore_sc = zeros(length(score_sc))
    for j in eachindex(score_sc)
        par1 = deepcopy(par)
        par1.scale[j] += ee
        ll1 = loglike(pm, par1)
        nscore_sc[j] = (ll1 - ll) / ee
    end
    @test isapprox(nscore_sc, score_sc, rtol = 1e-3, atol = 1e-3)

    # Check the score for smoothing parameters
    ee = 1e-6
    nscore_sm = zeros(length(score_sm))
    for j in eachindex(score_sm)
        par1 = deepcopy(par)
        par1.smooth[j] += ee
        ll1 = loglike(pm, par1)
        nscore_sm[j] = (ll1 - ll) / ee
    end
    @test isapprox(nscore_sm, score_sm, rtol = 1e-4, atol = 1e-4)

    # Check the score for the unexplained variance parameters
    ee = 1e-7
    nscore_ux = zeros(length(score_ux))
    for j in eachindex(score_ux)
        par1 = deepcopy(par)
        par1.unexplained[j] += ee
        ll1 = loglike(pm, par1)
        nscore_ux[j] = (ll1 - ll) / ee
    end
    @test isapprox(nscore_ux, score_ux, rtol = 1e-4, atol = 1e-4)
end

@testset "Check emulate" begin

    rng = StableRNG(123)

    n = 1000 # Number of groups
    m = 5    # Average number of observations per group

    par = GaussianParams([1.0, 0.0], [1.0, -1.5], [0.0, 0.0], [1.5, 0.0])

    ti = range(0, 1, length = m)
    tim = kron(ones(n), ti)
    x = zeros(n * m, 2)
    x[:, 1] .= 1
    x[:, 2] = tim
    X = Xmat(x, x, x, x)
    grp = kron(1:n, ones(m))
    pm = ProcessMLEModel(zeros(0), X, tim, grp; standardize = false)
    y = emulate(pm; par = par, rng = rng)
    f = 1.0
    pen = Penalty(zeros(0, 0), zeros(0, 0), Diagonal([f, f]), zeros(0, 0))
    pm1 = ProcessMLEModel(y, X, tim, grp; penalty = pen)
    fit!(pm1)

    # Regression tests
    par1 = pm1.params
    @test isapprox(par1.mean, [0.89, 0.25], rtol = 1e-2, atol = 1e-2)
    @test isapprox(par1.scale, [1.01, -1.52], rtol = 1e-2, atol = 1e-2)
    @test isapprox(par1.smooth, [0.06, 0.01], rtol = 1e-2, atol = 1e-2)
    @test isapprox(par1.unexplained, [1.47, 0.02], rtol = 1e-2, atol = 1e-2)
end

@testset "Check fitting (regularized)" begin

    rng = StableRNG(123)

    n = 1000     # Number of groups
    m = 5        # Number of observations per group
    nb = 5       # Number of basis functions for the mean structure
    bscale = 0.5 # Scale for basis functions

    # Population covariance matrix
    ti = collect(range(0, 1, length = m))
    cm = armat(ti)
    cmr = cholesky(cm)

    tim = kron(ones(n), ti)
    xb = get_basis(nb, tim, bscale)
    xa = ones(length(tim), 2)
    xa[:, 2] = tim
    X = Xmat(xb, xa, xa, xa)
    grp = kron((1:n), ones(m))

    # The mean structure
    ey = 0.4 * tim + (tim .- 0.5) .^ 2

    # Generate the response
    y = copy(ey)
    ii = 0
    for i = 1:n
        y[ii+1:ii+m] .+= cmr.L * randn(rng, m)
        ii += m
    end

    # Squared second derivative penalty
    a = collect(range(0, 1, length = 100))
    f2 = zeros(98, 100)
    for i = 1:98
        f2[i, i:i+2] = [1, -2, 1]
    end
    p0 = f2 * get_basis(nb, a, bscale)
    p0 = p0' * p0

    # Check that the fitted mean becomes smoother as the penalty
    # is strengthened
    d2a = []
    for f in [0, 1e3, 1e6]
        pen = Penalty(f * p0, zeros(0, 0), zeros(0, 0), zeros(0, 0))
        pm = ProcessMLEModel(y, X, tim, grp; penalty = pen)
        fit!(pm; verbose = false)

        yhat = get_basis(nb, a, bscale) * pm.params.mean
        d2 = diff(diff(yhat))
        push!(d2a, sum(abs2, d2))
    end

    @test maximum(diff(d2a)) < 0
end

@testset "Check fitting (simple)" begin

    rng = StableRNG(123)

    n = 1000 # Number of groups
    m = 5    # Average number of observations per group

    # Population covariance matrix
    ti = collect(range(0, 1, length = m))
    cm = armat(ti)
    cmr = cholesky(cm)

    x = zeros(n * m, 2)
    x[:, 1] .= 1
    x[:, 2] = kron(ones(n), ti)
    X = Xmat(x, x, x, x)

    tim = kron(ones(n), ti)
    grp = kron((1:n), ones(m))

    # The mean structure
    ey = x * [1, -1]

    # Generate the response
    y = copy(ey)
    ii = 0
    for i = 1:n
        y[ii+1:ii+m] .+= cmr.L * randn(rng, m)
        ii += m
    end

    pm1 = ProcessMLEModel(y, X, tim, grp; standardize = false)
    fit!(pm1; verbose = false)

    pm2 = ProcessMLEModel(y, X, tim, grp; standardize = true)
    fit!(pm2; verbose = false)

    par1 = pm1.params
    par2 = pm2.params

    @test isapprox(par1.mean, par2.mean)
    @test isapprox(par1.scale, par2.scale)
    @test isapprox(par1.smooth, par2.smooth)
    @test isapprox(par1.unexplained, par2.unexplained)

    @test isapprox(coef(pm1), coef(pm2), rtol = 1e-5, atol = 1e-5)
    @test isapprox(vcov(pm1), vcov(pm2), rtol = 1e-3, atol = 1e-3)

    x0 = ones(m, 2)
    x0[:, 2] = ti
    cpar = GaussianCovPar(x0 * par2.scale, x0 * par2.smooth, x0 * par2.unexplained)
    cm1 = covmat(cpar, x0[:, 2])
    cmd = cm - cm1
    @test maximum(abs.(cmd)) < 0.1
    @test maximum(abs.(cmd ./ cm)) < 0.1
end

@testset "Check covariance gradient" begin

    # Time points
    ti = [0.0, 1.0, 2.0, 3.0]

    # Smoothing and scale parameters at the given time points
    sm = [1.0, 2.0, 3.0, 4.0]
    sc = [5.0, 4.0, 3.0, 2.0]
    ux = [0.0, 0, 0, 0]

    # Covariance matrix and Jacobian at the given time points
    c = GaussianCovPar(sc, log.(sc), sm, log.(sm), ux, log.(ux))
    cc = covmat(c, ti)
    jsc, jsm = jac(c, ti)

    # Check the gradients using numeric differentiation.
    # The derivatives of cov with respect to sm[i] and
    # sc[i] are zero outside of column i so only compute
    # column i of the gradient.
    for i in eachindex(ti)
        # Check gradient with respect to scale
        ee = 1e-5
        sc1 = copy(sc)
        sc1[i] += ee
        c1 = GaussianCovPar(sc1, log.(sc1), sm, log.(sm), ux, log.(ux))
        cc1 = covmat(c1, ti)
        d = (cc1 .- cc) ./ ee
        @test isapprox(d[:, i], jsc[i], rtol = 1e-5, atol = 1e-5)

        # Check gradient with respect to smooth
        ee = 1e-5
        sm1 = copy(sm)
        sm1[i] += ee
        c1 = GaussianCovPar(sc, log.(sc), sm1, log.(sm1), ux, log.(ux))
        cc1 = covmat(c1, ti)
        d = (cc1 .- cc) ./ ee
        @test isapprox(d[:, i], jsm[i], rtol = 1e-5, atol = 1e-5)
    end
end
