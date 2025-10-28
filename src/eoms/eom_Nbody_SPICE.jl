"""Equations of motion"""


"""
    eom_Nbody_SPICE!(dx, x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_Nbody_SPICE!(dx, x, params, t)
    dx[1:3] = x[4:6]
    dx[4:6] = -params.mus[1] / norm(x[1:3])^3 * x[1:3]

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]
        else
            pos_3body, _ = spkpos(
                ID,
                params.et0 + t*params.TU,
                params.naif_frame,
                params.abcorr,
                params.naif_ids[1]
            )
            pos_3body /= params.DU
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
    eom_Nbody_SPICE(x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`
"""
function eom_Nbody_SPICE(x, params, t)
    dx = [x[4:6]; -params.mus[1] / norm(x[1:3])^3 * x[1:3]]

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]
        else
            pos_3body, _ = spkpos(
                ID,
                params.et0 + t*params.TU,
                params.naif_frame,
                params.abcorr,
                params.naif_ids[1]
            )
            pos_3body /= params.DU
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
    eom_stm_Nbody_SPICE!(dx_stm, x_stm, params, t)

Right-hand side of N-body equations of motion with STMcompatible with `DifferentialEquations.jl`
"""
function eom_stm_Nbody_SPICE!(dx_stm, x_stm, params, t)
    dx_stm[1:3] = x_stm[4:6]
    dx_stm[4:6] = -params.mus[1] / norm(x_stm[1:3])^3 * x_stm[1:3]

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]
        else
            pos_3body, _ = spkpos(
                ID,
                params.et0 + t*params.TU,
                params.naif_frame,
                params.abcorr,
                params.naif_ids[1]
            )
            pos_3body /= params.DU
        end

        if i >= 2
            params.Rs[1+3(i-2):3(i-1)] = pos_3body
            dx_stm[4:6] += third_body_accel(x_stm[1:3], pos_3body, mu_i)
        end

        if ID == "10" && params.include_srp
            params.R_sun = pos_3body
            dx_stm[4:6] += srp_cannonball(x_stm[1:3], pos_3body, params.k_srp_cannonball)
        end
    end

    if params.include_srp
        A = params.f_jacobian(x_stm[1:6], params.mus, params.Rs, params.k_srp_cannonball, params.R_sun)
    else
        A = params.f_jacobian(x_stm[1:6], params.mus, params.Rs)
    end
    dx_stm[7:42] = reshape((A * reshape(x_stm[7:42],6,6)')', 36)
    return nothing
end


"""
    dfdx_Nbody_SPICE(x, u, params, t)
    
Evaluate Jacobian of N-body problem
"""
function dfdx_Nbody_SPICE(x, u, params, t)
    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]
        else
            pos_3body, _ = spkpos(
                ID,
                params.et0 + t*params.TU,
                params.naif_frame,
                params.abcorr,
                params.naif_ids[1]
            )
            pos_3body /= params.DU
        end
        if i >= 2
            params.Rs[1+3(i-2):3(i-1)] = pos_3body
        end
        if ID == "10" && params.include_srp
            params.R_sun = pos_3body
        end
    end

    if params.include_srp
        return params.f_jacobian(x[1:6], params.mus, params.Rs, params.k_srp_cannonball, params.R_sun)
    else
        return params.f_jacobian(x[1:6], params.mus, params.Rs)
    end
end


"""
    eom_stm_Nbody_SPICE_fd!(dx_stm, x_stm, params, t)
    
Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`
"""
function eom_stm_Nbody_SPICE_fd!(dx_stm, x_stm, params, t)
    dx_stm[1:6] = eom_Nbody_SPICE(x_stm[1:6], params, t)
    A = eom_jacobian_fd(eom_Nbody_SPICE, x_stm[1:6], 0.0, params, t)
    A[1:3,4:6] .= I(3)   # force identity for linear map
    dx_stm[7:42] = reshape((A * reshape(x_stm[7:42],6,6)')', 36)
    return nothing
end