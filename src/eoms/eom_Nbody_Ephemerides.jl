"""Ephemerides.jl/FrameTransformations-based N-body equations of motion"""


"""
    eom_Nbody_Ephemerides!(dx, x, params, t)

Right-hand side of N-body equations of motion compatible with `DifferentialEquations.jl`.

Third-body positions are queried directly through `FrameTransformations.vector3`
using the Ephemerides-backed frame system in `params.ephemerides_frame_system`.
HFEM does not manually reconstruct missing ephemeris chains; missing kernel/frame
information should surface as backend errors.
"""
function eom_Nbody_Ephemerides!(dx, x, params, t)
    isnothing(params.ephemerides_frame_system) && error(
        "eom_Nbody_Ephemerides! requires `params.ephemerides_frame_system`. Pass `ephemerides_files`, `ephemerides_provider`, or `ephemerides_frame_system` to HighFidelityEphemerisModelParameters."
    )

    et = params.et0 + t * params.TU
    center_id = ephemerides_point_id(params.naif_ids[1])
    axes = ephemerides_axes_symbol(params.naif_frame)

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
                    axes,
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

    if params.include_drag
        isnothing(params.frame_PCPF) && error(
            "eom_Nbody_Ephemerides! requires `params.frame_PCPF` when drag is enabled."
        )
        T_inr2pcpf = Matrix(
            FrameTransformations.rotation3(
                params.ephemerides_frame_system,
                axes,
                ephemerides_axes_symbol(params.frame_PCPF),
                et,
            )[1]
        )
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
function eom_Nbody_Ephemerides(x, params, t)
    isnothing(params.ephemerides_frame_system) && error(
        "eom_Nbody_Ephemerides requires `params.ephemerides_frame_system`. Pass `ephemerides_files`, `ephemerides_provider`, or `ephemerides_frame_system` to HighFidelityEphemerisModelParameters."
    )

    et = params.et0 + t * params.TU
    center_id = ephemerides_point_id(params.naif_ids[1])
    axes = ephemerides_axes_symbol(params.naif_frame)

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
                    axes,
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

    if params.include_drag
        isnothing(params.frame_PCPF) && error(
            "eom_Nbody_Ephemerides requires `params.frame_PCPF` when drag is enabled."
        )
        T_inr2pcpf = Matrix(
            FrameTransformations.rotation3(
                params.ephemerides_frame_system,
                axes,
                ephemerides_axes_symbol(params.frame_PCPF),
                et,
            )[1]
        )
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
function eom_stm_Nbody_Ephemerides!(dx_stm, x_stm, params, t)
    if params.include_drag
        error("eom_stm_Nbody_Ephemerides! does not currently support drag.")
    end
    isnothing(params.ephemerides_frame_system) && error(
        "eom_stm_Nbody_Ephemerides! requires `params.ephemerides_frame_system`. Pass `ephemerides_files`, `ephemerides_provider`, or `ephemerides_frame_system` to HighFidelityEphemerisModelParameters."
    )

    et = params.et0 + t * params.TU
    center_id = ephemerides_point_id(params.naif_ids[1])
    axes = ephemerides_axes_symbol(params.naif_frame)

    dx_stm[1:3] = x_stm[4:6]
    dx_stm[4:6] = -params.mus[1] / norm(x_stm[1:3])^3 * x_stm[1:3]
    Rs = similar(params.Rs)
    R_sun = zeros(eltype(params.R_sun), length(params.R_sun))

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]   # needed in case Sun is central body (first body) & we need SRP
        else
            pos_3body = collect(
                FrameTransformations.vector3(
                    params.ephemerides_frame_system,
                    center_id,
                    ephemerides_point_id(ID),
                    axes,
                    et,
                )
            ) / params.DU
        end

        if i >= 2
            Rs[1+3(i-2):3(i-1)] = pos_3body
            dx_stm[4:6] += third_body_accel(x_stm[1:3], pos_3body, mu_i)
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
    dx_stm[7:42] = reshape((A * reshape(x_stm[7:42],6,6)), 36)
    return nothing
end


"""
    dfdx_Nbody_Ephemerides(x, u, params, t)

Evaluate Jacobian of N-body problem.
"""
function dfdx_Nbody_Ephemerides(x, u, params, t)
    if params.include_drag
        error("dfdx_Nbody_Ephemerides does not currently support drag.")
    end
    isnothing(params.ephemerides_frame_system) && error(
        "dfdx_Nbody_Ephemerides requires `params.ephemerides_frame_system`. Pass `ephemerides_files`, `ephemerides_provider`, or `ephemerides_frame_system` to HighFidelityEphemerisModelParameters."
    )

    et = params.et0 + t * params.TU
    center_id = ephemerides_point_id(params.naif_ids[1])
    axes = ephemerides_axes_symbol(params.naif_frame)

    Rs = similar(params.Rs)
    R_sun = zeros(eltype(params.R_sun), length(params.R_sun))

    for (i,(ID,mu_i)) in enumerate(zip(params.naif_ids, params.mus))
        if i == 1
            pos_3body = [0.0, 0.0, 0.0]   # needed in case Sun is central body (first body) & we need SRP
        else
            pos_3body = collect(
                FrameTransformations.vector3(
                    params.ephemerides_frame_system,
                    center_id,
                    ephemerides_point_id(ID),
                    axes,
                    et,
                )
            ) / params.DU
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
    eom_stm_Nbody_Ephemerides_fd!(dx_stm, x_stm, params, t)

Right-hand side of N-body equations of motion with STM compatible with `DifferentialEquations.jl`.
"""
function eom_stm_Nbody_Ephemerides_fd!(dx_stm, x_stm, params, t)
    dx_stm[1:6] = eom_Nbody_Ephemerides(x_stm[1:6], params, t)
    A = eom_jacobian_fd(eom_Nbody_Ephemerides, x_stm[1:6], 0.0, params, t)
    A[1:3,4:6] .= I(3)   # force identity for linear map
    dx_stm[7:42] = reshape((A * reshape(x_stm[7:42],6,6)), 36)
    return nothing
end