"""Atmospheric drag perturbation"""

"""
    get_drag_coefficient(DU, TU, VU, drag_Cd, drag_Am)

Get the drag coefficient `0.5 C_D A/m * 10^3 DU` in canonical units `[1/DU] / [kg/m^3]`

# Arguments
- `DU`: canonical distance unit, in km
- `TU`: canonical time unit, in s
- `VU`: canonical velocity unit, in km/s
- `drag_Cd`: drag coefficient, dimensionless
- `drag_Am`: area-to-mass ratio in units of `m^2/kg`
"""
function get_drag_coefficient(DU, TU, VU, drag_Cd, drag_Am)
    k_drag = 1e3 * DU * 0.5 * drag_Cd * drag_Am
    return k_drag
end


"""
    atmospheric_velocity(r_can, TU, omega_atm)

Compute co-rotating atmospheric velocity in canonical units.

# Arguments
- `r_can`: spacecraft position in canonical units
- `TU`: canonical time unit, in s
- `omega_atm`: atmospheric rotation rate in rad/s, in the inertial frame
"""
function atmospheric_velocity(r_can, TU, omega_atm)
    return TU * cross(omega_atm, r_can)
end


"""
    drag(r, v, v_atm, rho, k_drag)

Compute acceleration due to atmospheric drag in the inertial frame.

# Arguments
- `r`: position of the spacecraft, in canonical units (unused, kept for API symmetry)
- `v`: velocity of the spacecraft, in canonical units
- `v_atm`: velocity of the atmosphere, in canonical units
- `rho`: atmospheric density, in kg/DU^3
- `k_drag`: drag coefficient in canonical units
"""
function drag(r, v, v_atm, rho, k_drag)
    v_rel = v - v_atm
    v_rel_norm = norm(v_rel)
    return -k_drag * rho * v_rel_norm * v_rel
end


function _drag_accel_with_transform(x, params, t, T_inr2pcpf)
    et = params.et0 + t * params.TU
    r_km = T_inr2pcpf * x[1:3] * params.DU
    rho = params.f_density(et, r_km)
    v_atm = atmospheric_velocity(x[1:3], params.TU, params.omega_atm)
    return drag(x[1:3], x[4:6], v_atm, rho, params.k_drag)
end


function drag_accel_spice(x, params, t)
    et = params.et0 + t * params.TU
    T_inr2pcpf = SPICE.pxform(params.naif_frame, params.frame_PCPF, et)
    return _drag_accel_with_transform(x, params, t, T_inr2pcpf)
end


function drag_accel_interp(x, params, t)
    et = params.et0 + t * params.TU
    T_inr2pcpf = params.interpolated_transformation === nothing ?
        SPICE.pxform(params.naif_frame, params.frame_PCPF, et) :
        pxform(params.interpolated_transformation, et)
    return _drag_accel_with_transform(x, params, t, T_inr2pcpf)
end
