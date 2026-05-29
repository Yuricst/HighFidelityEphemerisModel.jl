"""Utility functions"""


"""
    mod_custom(a, n)

Custom modulo function
Ref: https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles
"""
function mod_custom(a, n)
    return a - floor(a / n) * n
end


"""
    angle_difference(ϕ_fwd::Real, ϕ_bck::Real)

Compute angle difference for periodic angles between 0 and 2π
"""
function angle_difference(ϕ_fwd::Real, ϕ_bck::Real)
    # modulo based
    dϕ = mod_custom((ϕ_bck - ϕ_fwd + π), 2π) - π
    return dϕ
end


"""
    vector_hessian_forwarddiff(f::Function, x)

Compute Hessian of a vector-valued function using ForwardDiff.
"""
function vector_hessian_forwarddiff(f::Function, x)
    out = ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x)
    return reshape(out, (length(x), length(x), length(x)))
end


"""
    eom_jacobian_fd(eom::Function, x, u, params, t)

Evaluate Jacobian of equations of motion using ForwardDiff.
The second argument `u` is a place-holder for control input.

# Arguments
- `eom`: "static" equations of motion, with signature `eom(x, params, t)` --> `dx`
- `x`: State vector
- `u`: Control input
- `params`: Parameters
- `t`: Time

# Returns
- `jac`: Jacobian of the equations of motion
"""
function eom_jacobian_fd(eom::Function, x, u, params, t)
    return ForwardDiff.jacobian(x -> eom(x, params, t), x)
end


"""
    eom_jacobian_centraldiff(eom::Function, x, u, params, t)

Evaluate Jacobian of equations of motion using central finite differences.
This is used for drag-enabled dynamics whose density callbacks need not accept
ForwardDiff dual numbers.
"""
function eom_jacobian_centraldiff(eom::Function, x, u, params, t; relstep::Float64 = cbrt(eps(Float64)))
    f0 = eom(x, params, t)
    jac = zeros(eltype(f0), length(f0), length(x))

    for i in eachindex(x)
        h = relstep * max(abs(Float64(x[i])), 1.0)
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        jac[:, i] = (eom(x_plus, params, t) - eom(x_minus, params, t)) / (2h)
    end

    return jac
end


function eom_jacobian_drag_safe(eom::Function, x, u, params, t)
    if params.include_drag
        return eom_jacobian_centraldiff(eom, x, u, params, t)
    end

    return eom_jacobian_fd(eom, x, u, params, t)
end


"""
    eom_hessian_fd(eom::Function, x, u, params, t)

Evaluate Hessian of equations of motion using ForwardDiff.
The second argument `u` is a place-holder for control input.

# Arguments
- `eom`: "static" equations of motion, with signature `eom(x, params, t)` --> `dx`
- `x`: State vector
- `u`: Control input
- `params`: Parameters
- `t`: Time

# Returns
- `hess`: Hessian of the equations of motion
"""
function eom_hessian_fd(eom::Function, x, u, params, t)
    return vector_hessian_forwarddiff(x -> eom(x, params, t), x)
end


function eom_jacobian_sparsediff(eom::Function, x, u, params, t)
    return sparse_jacobian(params.adtype, params.jacobian_cache, eom, x -> eom(x, params, t))
end


function factorial_safe(n::Int)
    if n <= 20
        return factorial(n)
    else
        return factorial(big(n))
    end
end
