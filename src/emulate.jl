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
        while true
            tt = 2 .* rand(b) .- 1
            sort!(tt)
            if b == 1 || minimum(diff(tt)) > 0.5 / (b - 1)
                push!(ti, tt...)
                break
            end
        end
    end

    return tuple(ti, grp, ix)
end

#=
    emulate(par; ...)

Simulate data from a fitted ProcessRegression model.
=#
function emulate(pm::ProcessMLEModel; par = nothing)
    par = isnothing(par) ? pm.params : par
    y_mean = pm.X.mean * par.mean
    grp = pm.grp

    # Simulate the response values
    y = copy(y_mean)
    for (i1, i2) in eachcol(grp)
        cpar = covpar(pm, par, i1, i2)
        cm = covmat(cpar, pm.time[i1:i2])
        cr = cholesky(cm)
        y[i1:i2] .+= cr.U' * randn(i2 - i1 + 1)
    end

    return y
end
