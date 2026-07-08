"""Ephemerides.jl/FrameTransformations-based N-body equations of motion"""


@inline function _assert_ephemerides_abcorr_supported(params::EphemeridesParameters)
    params.abcorr == "NONE" || _validate_ephemerides_abcorr(params.abcorr)
    return nothing
end


function _get_pos_3body_ephemerides(params::EphemeridesParameters, ID::String, t)
    et = params.et0 + t * params.TU
    backend = ephemerides_backend(params)

    return collect(
        get_pos_ephemerides(
            backend.frame_system,
            ID,
            params.naif_ids[1],
            et;
            axes = params.naif_frame,
        )
    ) / params.DU
end


"""
    eom_Nbody_Ephemerides!(dx, x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`.

Third-body positions are queried directly through `FrameTransformations.vector3`
using the Ephemerides-backed frame system in `params.ephemerides_frame_system`.
HFEM uses the Ephemerides backend attached to `params`.
"""
function eom_Nbody_Ephemerides!(dx, x, params::EphemeridesParameters, t)
    _assert_ephemerides_abcorr_supported(params)

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

    if params.include_drag
        isnothing(params.frame_PCPF) && error(
            "eom_Nbody_Ephemerides! requires `params.frame_PCPF` when drag is enabled."
        )
        et = params.et0 + t * params.TU
        T_inr2pcpf = pxform_ephemerides(params, params.naif_frame, params.frame_PCPF, et)
        r_km = T_inr2pcpf * x[1:3] * params.DU
        rho = params.f_density(et, r_km)
        v_atm = atmospheric_velocity(x[1:3], params.TU, params.omega_atm)
        dx[4:6] += drag(x[1:3], x[4:6], v_atm, rho, params.k_drag)
    end

    return nothing
end


"""
    eom_Nbody_Ephemerides(x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`.
"""
function eom_Nbody_Ephemerides(x, params::EphemeridesParameters, t)
    _assert_ephemerides_abcorr_supported(params)

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

    if params.include_drag
        isnothing(params.frame_PCPF) && error(
            "eom_Nbody_Ephemerides requires `params.frame_PCPF` when drag is enabled."
        )
        et = params.et0 + t * params.TU
        T_inr2pcpf = pxform_ephemerides(params, params.naif_frame, params.frame_PCPF, et)
        r_km = T_inr2pcpf * x[1:3] * params.DU
        rho = params.f_density(et, r_km)
        v_atm = atmospheric_velocity(x[1:3], params.TU, params.omega_atm)
        dx[4:6] += drag(x[1:3], x[4:6], v_atm, rho, params.k_drag)
    end

    return dx
end


"""
    eom_stm_Nbody_Ephemerides!(dx_stm, x_stm, params, t)

Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`.
"""
function eom_stm_Nbody_Ephemerides!(dx_stm, x_stm, params::EphemeridesParameters, t)
    return eom_stm_Nbody_Ephemerides_fd!(dx_stm, x_stm, params, t)
end


"""
    eom_stm_Nbody_Ephemerides_fd!(dx_stm, x_stm, params, t)

Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`.
"""
function eom_stm_Nbody_Ephemerides_fd!(dx_stm, x_stm, params::EphemeridesParameters, t)
    dx_stm[1:6] = eom_Nbody_Ephemerides(x_stm[1:6], params, t)
    A = eom_jacobian_fd(eom_Nbody_Ephemerides, x_stm[1:6], 0.0, params, t)
    A[1:3,4:6] .= I(3)   # force identity for linear map
    dx_stm[7:42] = reshape((A * reshape(x_stm[7:42],6,6)), 36)
    return nothing
end