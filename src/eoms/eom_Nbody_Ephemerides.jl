"""Ephemerides.jl-based N-body equations of motion"""


function _get_pos_3body_ephemerides(params, ID::String, t)
    return collect(
        get_pos_ephemerides(
            params.ephemerides_provider,
            ID,
            params.naif_ids[1],
            params.et0 + t*params.TU,
        )
    ) / params.DU
end


"""
    eom_Nbody_Ephemerides!(dx, x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_Nbody_Ephemerides!(dx, x, params, t)
    if params.include_drag
        error("eom_Nbody_Ephemerides! does not currently support drag.")
    end

    dx[1:3] = x[4:6]
    dx[4:6] = -params.mus[1] / norm(x[1:3])^3 * x[1:3]

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]   # needed in case Sun is central body (first body) & we need SRP
        else
            pos_3body = _get_pos_3body_ephemerides(params, ID, t)
        end

        if i >= 2
            dx[4:6] += third_body_accel(x[1:3], pos_3body, mu_i)
        end

        if ID == "10" && params.include_srp
            dx[4:6] += srp_cannonball(x[1:3], pos_3body, params.k_srp_cannonball)
        end
    end

    return nothing
end


"""
    eom_Nbody_Ephemerides(x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_Nbody_Ephemerides(x, params, t)
    if params.include_drag
        error("eom_Nbody_Ephemerides does not currently support drag.")
    end

    dx = [x[4:6]; -params.mus[1] / norm(x[1:3])^3 * x[1:3]]

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]   # needed in case Sun is central body (first body) & we need SRP
        else
            pos_3body = _get_pos_3body_ephemerides(params, ID, t)
        end

        if i >= 2
            dx[4:6] += third_body_accel(x[1:3], pos_3body, mu_i)
        end

        if ID == "10" && params.include_srp
            dx[4:6] += srp_cannonball(x[1:3], pos_3body, params.k_srp_cannonball)
        end
    end

    return dx
end


"""
    eom_stm_Nbody_Ephemerides_fd!(dx_stm, x_stm, params, t)

Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`
"""
function eom_stm_Nbody_Ephemerides_fd!(dx_stm, x_stm, params, t)
    dx_stm[1:6] = eom_Nbody_Ephemerides(x_stm[1:6], params, t)
    A = eom_jacobian_fd(eom_Nbody_Ephemerides, x_stm[1:6], 0.0, params, t)
    A[1:3,4:6] .= I(3)   # force identity for linear map
    dx_stm[7:42] = reshape((A * reshape(x_stm[7:42],6,6)), 36)
    return nothing
end
