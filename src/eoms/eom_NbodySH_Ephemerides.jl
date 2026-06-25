"""Ephemerides.jl-based N-body equations of motion with spherical harmonics"""


"""
    eom_NbodySH_Ephemerides!(dx, x, params, t)

Right-hand side of N-body equations of motion with spherical harmonics compatible
with `DifferentialEquations.jl`.

Third-body positions are queried through Ephemerides.jl. Frame transformations for
spherical harmonics and drag are queried through FrameTransformations.jl via
`pxform_ephemerides`.
"""
function eom_NbodySH_Ephemerides!(dx, x, params, t)
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

    et = params.et0 + t * params.TU

    if params.include_drag
        T_inr2pcpf = pxform_ephemerides(params, params.naif_frame, params.frame_PCPF, et)
        r_km = T_inr2pcpf * x[1:3] * params.DU
        rho = params.f_density(et, r_km)
        v_atm = atmospheric_velocity(x[1:3], params.TU, params.omega_atm)
        dx[4:6] += drag(x[1:3], x[4:6], v_atm, rho, params.k_drag)
    end

    if !isnothing(params.spherical_harmonics_data)
        T_inr2pcpf = pxform_ephemerides(params, params.naif_frame, params.frame_PCPF, et)
        a_SH = spherical_harmonics_accel(
            T_inr2pcpf,
            x[1:3] * params.DU,
            params.spherical_harmonics_data["Cnm"],
            params.spherical_harmonics_data["Snm"],
            params.spherical_harmonics_data["GM"],
            params.spherical_harmonics_data["REFERENCE RADIUS"],
            params.spherical_harmonics_data["nmax"],
            params.factorial_alias,
        )
        dx[4:6] += a_SH / (params.VU/params.TU)
    end

    return nothing
end


"""
    eom_NbodySH_Ephemerides(x, params, t)

Right-hand side of N-body equations of motion with spherical harmonics compatible
with `DifferentialEquations.jl`.
"""
function eom_NbodySH_Ephemerides(x, params, t)
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

    et = params.et0 + t * params.TU

    if params.include_drag
        T_inr2pcpf = pxform_ephemerides(params, params.naif_frame, params.frame_PCPF, et)
        r_km = T_inr2pcpf * x[1:3] * params.DU
        rho = params.f_density(et, r_km)
        v_atm = atmospheric_velocity(x[1:3], params.TU, params.omega_atm)
        dx[4:6] += drag(x[1:3], x[4:6], v_atm, rho, params.k_drag)
    end

    if !isnothing(params.spherical_harmonics_data)
        T_inr2pcpf = pxform_ephemerides(params, params.naif_frame, params.frame_PCPF, et)
        a_SH = spherical_harmonics_accel(
            T_inr2pcpf,
            x[1:3] * params.DU,
            params.spherical_harmonics_data["Cnm"],
            params.spherical_harmonics_data["Snm"],
            params.spherical_harmonics_data["GM"],
            params.spherical_harmonics_data["REFERENCE RADIUS"],
            params.spherical_harmonics_data["nmax"],
            params.factorial_alias,
        )
        dx[4:6] += a_SH / (params.VU/params.TU)
    end

    return dx
end


"""
    dfdx_NbodySH_Ephemerides_fd(x, u, params, t)

Evaluate Jacobian of N-body problem with spherical harmonics using ForwardDiff.
"""
function dfdx_NbodySH_Ephemerides_fd(x, u, params, t)
    return ForwardDiff.jacobian(x -> HighFidelityEphemerisModel.eom_NbodySH_Ephemerides(x, params, t), x)
end


"""
    eom_stm_NbodySH_Ephemerides_fd!(dx_stm, x_stm, params, t)

Right-hand side of N-body equations of motion with spherical harmonics and STM
compatible with `DifferentialEquations.jl`.
"""
function eom_stm_NbodySH_Ephemerides_fd!(dx_stm, x_stm, params, t)
    dx_stm[1:6] = eom_NbodySH_Ephemerides(x_stm[1:6], params, t)
    A = eom_jacobian_fd(eom_NbodySH_Ephemerides, x_stm[1:6], 0.0, params, t)
    A[1:3,4:6] .= I(3)   # force identity for linear map
    dx_stm[7:42] = reshape((A * reshape(x_stm[7:42],6,6)), 36)
    return nothing
end
