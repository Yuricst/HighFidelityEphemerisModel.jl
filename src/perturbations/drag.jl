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