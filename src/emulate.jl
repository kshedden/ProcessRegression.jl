function genx(N, fix_unexplained, par)

    # Design matrices
    X_mean = randn(N, length(par.mean))
    X_scale = randn(N, length(par.scale))
    X_smooth = randn(N, length(par.smooth))
    X_unexplained = randn(
        N,
        length(fix_unexplained) == 0 ? length(par.unexplained) : length(fix_unexplained),
    )

    # Always include an intercept
    X_mean[:, 1] .= 1
    X_scale[:, 1] .= 1
    X_smooth[:, 1] .= 1
    X_unexplained[:, 1] .= 1

    return Xmat(X_mean, X_scale, X_smooth, X_unexplained)
end

function gengrptime(n, m)
    grp = Int[]
    ti = Float64[]
    ix = zeros(Int, 2, n)
    ii = 1
    for i = 1:n
        b = Int64(ceil(m * rand())) # Group size
        push!(grp, fill(i, b)...)
        ix[:, i] = [ii, ii + b - 1]
        ii += b
        push!(ti, (2 .* rand(b) .- 1)...)
    end
    return tuple(ti, grp, ix)
end

#=
    emulate(par; ...)

Simulate data from a fitted ProcessRegression model.
=#
function emulate(
    par;
    n = nothing,
    m = nothing,
    time = nothing,
    X = nothing,
    grp = nothing,
    fix_unexplained = zeros(0),
)

    if X == nothing && (n == nothing || m == nothing)
        error("Either X or both n and m must be specified")
    end

    # Simulate group and/or time if not provided
    if grp == nothing || time == nothing
        tim_, grp_, ix = gengrptime(n, m)
    else
        ix, _ = groupix(grp)
    end
    grp = grp == nothing ? grp_ : grp
    tim = time == nothing ? tim_ : time

    # Total number of observations
    N = length(grp)

    # Simulate the explanatory variables if not provided
    X = X == nothing ? genx(N, fix_unexplained, par) : X

    # Mean values
    y_mean = X.mean * par.mean

    # Placeholder
    yy = zeros(Float64, N)

    pm = ProcessMLEModel(yy, X, tim, grp; fix_unexplained = copy(fix_unexplained))

    # Simulate the response values
    y = copy(y_mean)
    for (i1, i2) in eachcol(ix)
        cpar = covpar(pm, par, i1, i2)
        cm = covmat(cpar, tim[i1:i2])
        cr = cholesky(cm)
        y[i1:i2] .+= cr.U' * randn(i2 - i1 + 1)
    end

    # Replace with the actual values
    pm.y = y

    return pm
end
