using ProcessRegression, Test, LinearAlgebra

@testset "Check fitting" begin

    n = 1000 # Number of groups
    m = 5    # Average number of observations per group

    for fix_unexplained in [zeros(0), [0.1]]

        par = GaussianParams(
            [1.0, -1.5],
            [2.0, 0.0],
            [1.5, 0.0],
            length(fix_unexplained) > 0 ? fix_unexplained : [1.0, 0],
        )

        # Fit once with a penalty
        pm = emulate(par; n = n, m = m, fix_unexplained = fix_unexplained)
        pm.penalty = Penalty(10, 10, 10, 0)
        fit!(pm; verbose = false)
        coef(pm)
        println(coeftable(pm))

        # Use the penalized results as starting values
        # for an unpenalized fit
        pm.penalty = Penalty(0, 0, 0, 0)
        fit!(pm; start = pm.params, verbose = false)
        coef(pm)
        println(coeftable(pm))
    end
end

@testset "Check likelihood gradient using numerical derivatives" begin

    par = GaussianParams([1.0, -1.5], [2.0, 0.0], [1.5, 0.0], [1.0, 0])

    n = 1000 # Number of groups
    m = 5    # Average number of observations per group

    pm = emulate(par; n = n, m = m)

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

@testset "Check covariance gradient" begin

    # Time points
    ti = [0.0, 1.0, 2.0, 3.0]

    # Smoothing and scale parameters at the given time points
    sm = [1.0, 2.0, 3.0, 4.0]
    sc = [5.0, 4.0, 3.0, 2.0]
    ux = [0.0, 0, 0, 0]

    # Covariance matrix and Jacobian at the given time points
    c = GaussianCovPar(sc, sm, ux)
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
        c1 = GaussianCovPar(sc1, sm, ux)
        cc1 = covmat(c1, ti)
        d = (cc1 .- cc) ./ ee
        @test isapprox(d[:, i], jsc[i], rtol = 1e-5, atol = 1e-5)

        # Check gradient with respect to smooth
        ee = 1e-5
        sm1 = copy(sm)
        sm1[i] += ee
        c1 = GaussianCovPar(sc, sm1, ux)
        cc1 = covmat(c1, ti)
        d = (cc1 .- cc) ./ ee
        @test isapprox(d[:, i], jsm[i], rtol = 1e-5, atol = 1e-5)

    end

end
