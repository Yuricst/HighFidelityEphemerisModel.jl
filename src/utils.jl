"""Utility functions"""


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