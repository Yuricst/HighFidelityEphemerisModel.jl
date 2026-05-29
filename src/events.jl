"""Event functions"""


"""
Compute osculating true anomaly from state and mu
    
# Arguments
- `state::Vector`: state vector in Cartesian coordinates, in order [x, y, z, vx, vy, vz]
- `mu::Float64`: gravitational parameter
- `to2pi::Bool`: if true, return the true anomaly in the range [0, 2π]
"""
function cart2trueanomaly(state::Vector, mu::Float64; to2pi::Bool=false)
    r = state[1:3]
    v = state[4:6]
    h = cross(r,v)
    hnorm = norm(h)
    vr = dot(v,r)/norm(r)
    ta = atan(hnorm*vr, hnorm^2/norm(r) - mu)
    if to2pi == true
        return mod(ta, 2π)
    else
        return ta
    end
end


"""
Get event function to detect target osculating true anomaly

# Arguments
- `θ_target::Real`: target osculating true anomaly
- `t_bounds::Tuple{Real,Real}`: time bounds
- `radius_bounds::Tuple{Real,Real}`: radius bounds
- `θ_check_range::Float64`: check range for true anomaly

# Returns:
- `_condition`: event function with signature `(x, t, integrator) -> Union{Float64, NaN}`
"""
function get_trueanomaly_event(
    θ_target::Real;
    t_bounds::Tuple{Real,Real} = (-1e12, 1e12), 
    radius_bounds::Tuple{Real,Real} = (0.0, 1e12),
    θ_check_range::Float64 = deg2rad(60),
)
    if θ_target >= 0.9 * π
        to2pi = true
    else
        to2pi = false
    end

    function _condition(x, t, integrator)
        rnorm = norm(x[1:3])
        if (radius_bounds[1] <= rnorm <= radius_bounds[2]) && (t_bounds[1] <= t <= t_bounds[2])
            θ = cart2trueanomaly(x[1:6], integrator.p.mus[1]; to2pi = to2pi)
            if abs(θ - θ_target) < θ_check_range
                return angle_difference(θ, θ_target)
            else
                return NaN
            end
        else
            return NaN
        end
    end
    return _condition
end