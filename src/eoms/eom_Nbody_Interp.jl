"""Interpolated ephemeris-based N-body equations of motion"""


"""
    eom_Nbody_Interp!(dx, x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_Nbody_Interp!(dx, x, params, t)
    dx[1:3] = x[4:6]
    dx[4:6] = -params.mus[1] / norm(x[1:3])^3 * x[1:3]

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]   # needed in case Sun is central body (first body) & we need SRP
        else
            pos_3body = get_pos(params.interpolated_ephems[i-1], params.et0 + t*params.TU) /params.DU
        end

        if i >= 2
            dx[4:6] += third_body_accel(x[1:3], pos_3body, mu_i)
        end

        if ID == "10" && params.include_srp
            dx[4:6] += srp_cannonball(x[1:3], pos_3body, params.k_srp_cannonball)
        end
    end

    if params.include_drag
        et = params.et0 + t * params.TU
        r_km = x[1:3] * params.DU
        rho = params.f_density(et, r_km)
        v_atm = atmospheric_velocity(x[1:3], params.TU, params.omega_atm)
        dx[4:6] += drag(x[1:3], x[4:6], v_atm, rho, params.k_drag)
    end
    return nothing
end


"""
    eom_Nbody_Interp(x, params, t)
    
Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_Nbody_Interp(x, params, t)
    dx = [x[4:6]; -params.mus[1] / norm(x[1:3])^3 * x[1:3]]
    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]   # needed in case Sun is central body (first body) & we need SRP
        else
            pos_3body = get_pos(params.interpolated_ephems[i-1], params.et0 + t*params.TU) /params.DU
        end

        if i >= 2
            dx[4:6] += third_body_accel(x[1:3], pos_3body, mu_i)
        end

        if ID == "10" && params.include_srp
            dx[4:6] += srp_cannonball(x[1:3], pos_3body, params.k_srp_cannonball)
        end
    end

    if params.include_drag
        et = params.et0 + t * params.TU
        r_km = x[1:3] * params.DU
        rho = params.f_density(et, r_km)
        v_atm = atmospheric_velocity(x[1:3], params.TU, params.omega_atm)
        dx[4:6] += drag(x[1:3], x[4:6], v_atm, rho, params.k_drag)
    end
    return dx
end


"""
    eom_stm_Nbody_Interp!(dx_stm, x_stm, params, t)
    
Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`
"""
function eom_stm_Nbody_Interp!(dx_stm, x_stm, params, t)
    if params.include_drag
        dx_stm[1:6] = eom_Nbody_Interp(x_stm[1:6], params, t)
        A = eom_jacobian_centraldiff(eom_Nbody_Interp, x_stm[1:6], 0.0, params, t)
        A[1:3,4:6] .= I(3)
        dx_stm[7:42] = reshape((A * reshape(x_stm[7:42],6,6)')', 36)
        return nothing
    end

    dx_stm[1:3] = x_stm[4:6]
    dx_stm[4:6] = -params.mus[1] / norm(x_stm[1:3])^3 * x_stm[1:3]
    Rs = similar(params.Rs)
    R_sun = zeros(eltype(params.R_sun), length(params.R_sun))

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]   # needed in case Sun is central body (first body) & we need SRP
        else
            pos_3body = get_pos(params.interpolated_ephems[i-1], params.et0 + t*params.TU) /params.DU
        end

        if i >= 2
            Rs[1+3(i-2):3(i-1)] = pos_3body
            dx_stm[4:6] += third_body_accel(x_stm[1:3], pos_3body, params.mus[i])
        end

        if ID == "10" && params.include_srp
            R_sun .= pos_3body
            dx_stm[4:6] += srp_cannonball(x_stm[1:3], pos_3body, params.k_srp_cannonball)
        end
    end

    if params.include_srp
        A = params.f_jacobian(x_stm[1:6], params.mus, Rs, params.k_srp_cannonball, R_sun)
    else
        A = params.f_jacobian(x_stm[1:6], params.mus, Rs)
    end
    dx_stm[7:42] = reshape((A * reshape(x_stm[7:42],6,6)')', 36)
    return nothing
end


"""
    dfdx_Nbody_Interp(x, u, params, t)
    
Evaluate Jacobian of N-body problem
"""
function dfdx_Nbody_Interp(x, u, params, t)
    if params.include_drag
        return eom_jacobian_centraldiff(eom_Nbody_Interp, x[1:6], u, params, t)
    end

    Rs = similar(params.Rs)
    R_sun = zeros(eltype(params.R_sun), length(params.R_sun))

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]   # needed in case Sun is central body (first body) & we need SRP
        else
            pos_3body = get_pos(params.interpolated_ephems[i-1], params.et0 + t*params.TU) /params.DU
        end
        if i >= 2
            Rs[1+3(i-2):3(i-1)] = pos_3body
        end
        if ID == "10" && params.include_srp
            R_sun .= pos_3body
        end
    end
    
    if params.include_srp
        return params.f_jacobian(x[1:6], params.mus, Rs, params.k_srp_cannonball, R_sun)
    else
        return params.f_jacobian(x[1:6], params.mus, Rs)
    end
end


"""
    eom_stm_Nbody_Interp_fd!(dx_stm, x_stm, params, t)
    
Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`
"""
function eom_stm_Nbody_Interp_fd!(dx_stm, x_stm, params, t)
    dx_stm[1:6] = eom_Nbody_Interp(x_stm[1:6], params, t)
    A = eom_jacobian_drag_safe(eom_Nbody_Interp, x_stm[1:6], 0.0, params, t)
    A[1:3,4:6] .= I(3)   # force identity for linear map
    dx_stm[7:42] = reshape((A * reshape(x_stm[7:42],6,6)')', 36)
    return nothing
end

