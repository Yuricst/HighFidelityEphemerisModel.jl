"""Atmospheric drag perturbation"""

"""
    get_drag_coefficient(DU, TU, VU, drag_Cd, drag_Am)

Get the drag coefficient `0.5 C_D A/m * 10^3 DU` in canonical units `[1/DU] / [kg/m^3]`

# Arguments
- `DU`: canonical distance unit, in km
- `TU`: canonical time unit, in s (unused, kept for API symmetry)
- `VU`: canonical velocity unit, in km/s (unused, kept for API symmetry)
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
- `rho`: atmospheric density, in kg/m^3 (unit conversion is folded into `k_drag`)
- `k_drag`: drag coefficient in canonical units
"""
function drag(r, v, v_atm, rho, k_drag)
    v_rel = v - v_atm
    v_rel_norm = norm(v_rel)
    return -k_drag * rho * v_rel_norm * v_rel
end

"""
    _drag3(vx, vy, vz, rx, ry, rz, rho, TU, omega_atm, k_drag)

Allocation-free scalar form of `drag(r, v, atmospheric_velocity(r, TU, omega_atm), rho, k_drag)`.
Returns the canonical drag acceleration as a 3-tuple.
"""
@inline function _drag3(vx, vy, vz, rx, ry, rz, rho, TU, omega_atm, k_drag)
    wx, wy, wz = omega_atm[1], omega_atm[2], omega_atm[3]
    vrx = vx - TU * (wy * rz - wz * ry)
    vry = vy - TU * (wz * rx - wx * rz)
    vrz = vz - TU * (wx * ry - wy * rx)
    c = -k_drag * rho * sqrt(vrx * vrx + vry * vry + vrz * vrz)
    return (c * vrx, c * vry, c * vrz)
end
