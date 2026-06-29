"""Ephemerides.jl/FrameTransformations-based N-body equations of motion with spherical harmonics"""


"""
    eom_NbodySH_Ephemerides!(dx, x, params, t)

Right-hand side of N-body equations of motion with spherical harmonics compatible
with `DifferentialEquations.jl`.

Third-body positions are queried through `FrameTransformations.vector3` and frame
rotations are queried through `FrameTransformations.rotation3` using the
Ephemerides-backed frame system in `params.ephemerides_frame_system`.
"""
function eom_NbodySH_Ephemerides!(dx, x, params, t)
    isnothing(params.ephemerides_frame_system) && error(
        "eom_NbodySH_Ephemerides! requires `params.ephemerides_frame_system`. Pass `ephemerides_files`, `ephemerides_provider`, or `ephemerides_frame_system` to HighFidelityEphemerisModelParameters."
    )

    et = params.et0 + t * params.TU
    center_id = ephemerides_point_id(params.naif_ids[1])
    axes_inr = ephemerides_axes_symbol(params.naif_frame)

    dx[1:3] = x[4:6]
    dx[4:6] = -params.mus[1] / norm(x[1:3])^3 * x[1:3]

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]   # needed in case Sun is central body (first body) & we need SRP
        else
            pos_3body = collect(
                FrameTransformations.vector3(
                    params.ephemerides_frame_system,
                    center_id,
                    ephemerides_point_id(ID),
                    axes_inr,
                    et,
                )
            ) / params.DU
        end

        if i >= 2
            dx[4:6] += third_body_accel(x[1:3], pos_3body, mu_i)
        end

        if ID == "10" && params.include_srp
            dx[4:6] += srp_cannonball(x[1:3], pos_3body, params.k_srp_cannonball)
        end
    end

    T_inr2pcpf = nothing
    if params.include_drag || !isnothing(params.spherical_harmonics_data)
        isnothing(params.frame_PCPF) && error(
            "eom_NbodySH_Ephemerides! requires `params.frame_PCPF` when drag or spherical harmonics are enabled."
        )
        T_inr2pcpf = Matrix(
            FrameTransformations.rotation3(
                params.ephemerides_frame_system,
                axes_inr,
                ephemerides_axes_symbol(params.frame_PCPF),
                et,
            )[1]
        )
    end

    if params.include_drag
        r_km = T_inr2pcpf * x[1:3] * params.DU
        rho = params.f_density(et, r_km)
        v_atm = atmospheric_velocity(x[1:3], params.TU, params.omega_atm)
        dx[4:6] += drag(x[1:3], x[4:6], v_atm, rho, params.k_drag)
    end

    if !isnothing(params.spherical_harmonics_data)
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
    isnothing(params.ephemerides_frame_system) && error(
        "eom_NbodySH_Ephemerides requires `params.ephemerides_frame_system`. Pass `ephemerides_files`, `ephemerides_provider`, or `ephemerides_frame_system` to HighFidelityEphemerisModelParameters."
    )

    et = params.et0 + t * params.TU
    center_id = ephemerides_point_id(params.naif_ids[1])
    axes_inr = ephemerides_axes_symbol(params.naif_frame)

    dx = [x[4:6]; -params.mus[1] / norm(x[1:3])^3 * x[1:3]]

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]   # needed in case Sun is central body (first body) & we need SRP
        else
            pos_3body = collect(
                FrameTransformations.vector3(
                    params.ephemerides_frame_system,
                    center_id,
                    ephemerides_point_id(ID),
                    axes_inr,
                    et,
                )
            ) / params.DU
        end

        if i >= 2
            dx[4:6] += third_body_accel(x[1:3], pos_3body, mu_i)
        end

        if ID == "10" && params.include_srp
            dx[4:6] += srp_cannonball(x[1:3], pos_3body, params.k_srp_cannonball)
        end
    end

    T_inr2pcpf = nothing
    if params.include_drag || !isnothing(params.spherical_harmonics_data)
        isnothing(params.frame_PCPF) && error(
            "eom_NbodySH_Ephemerides requires `params.frame_PCPF` when drag or spherical harmonics are enabled."
        )
        T_inr2pcpf = Matrix(
            FrameTransformations.rotation3(
                params.ephemerides_frame_system,
                axes_inr,
                ephemerides_axes_symbol(params.frame_PCPF),
                et,
            )[1]
        )
    end

    if params.include_drag
        r_km = T_inr2pcpf * x[1:3] * params.DU
        rho = params.f_density(et, r_km)
        v_atm = atmospheric_velocity(x[1:3], params.TU, params.omega_atm)
        dx[4:6] += drag(x[1:3], x[4:6], v_atm, rho, params.k_drag)
    end

    if !isnothing(params.spherical_harmonics_data)
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