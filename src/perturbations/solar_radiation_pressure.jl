"""Solar radiation pressure"""


"""
    get_srp_cannonball_coefficient(srp_P0, srp_Cr, srp_Am, DU, TU; AU = 149.6e6)

Get the cannonball coefficient in canonical units for solar radiation pressure.

# Arguments
- `DU`: canonical distance unit, in km
- `TU`: canonical time unit, in s
- `srp_Cr`: reflectivity coefficient, dimensionless
- `srp_Am`: area-to-mass ratio in units of `m^2/kg`
- `srp_P0`: solar radiation pressure at 1 AU, in units of `N/m^2`
- `AU`: Astronomical unit, in km
"""
function get_srp_cannonball_coefficient(
    DU,
    TU,
    srp_Cr,
    srp_Am,
    srp_P0 = 4.56e-6;
    AU = 149597870.7,
)
    k_srp_cannonball = (AU/DU)^2 * (srp_P0 * srp_Cr * srp_Am / 1000) * (TU^2/DU)
    return k_srp_cannonball
end


"""
    srp_cannonball(r, r_sun, k_srp_cannonball)

Compute acceleration due to solar radiation pressure using the cannonball model.

# Arguments
- `r`: position of the spacecraft, in km
- `r_sun`: position of the sun, in km
- `k_srp_cannonball`: cannonball coefficient in canonical units DU^3/TU^2
"""
function srp_cannonball(r::Vector{T}, r_sun::Vector{Float64}, k_srp_cannonball::Float64) where T
    r_relative = r - r_sun
    return k_srp_cannonball * r_relative / norm(r_relative)^3
end