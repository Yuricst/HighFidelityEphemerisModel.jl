"""Harris-Priester atmospheric density model"""


"""Interpolate look-up table for Harris-Priester model for atmospheric density.

Columns are:
   h [km], rho_min [g/km^3]k, rho_max [g/km^3]
Interpolation is done based on input h [km],
and returns density in [g/km^3].

Conversions of density:
To convert from [g/km^3] to [kg/m^3], multiply by 1e12.
"""
const _HARRIS_PRIESTER_DATA = [
    100 497400.0 497400.0
    120 24900.0 24900.0
    130 8377.0 8710.0
    140 3899.0 4059.0
    150 2122.0 2215.0
    160 1263.0 1344.0
    170 800.8 875.8
    180 528.3 601.0
    190 361.7 429.7
    200 255.7 316.2
    210 183.9 239.6
    220 134.1 185.3
    230 99.49 145.5
    240 74.88 115.7
    250 57.09 93.08
    260 44.03 75.55
    270 34.30 61.82
    280 26.97 50.95
    290 21.39 42.26
    300 17.08 35.26
    320 10.99 25.11
    340 7.214 18.19
    360 4.824 13.37
    380 3.274 9.955
    400 2.249 7.492
    420 1.558 5.684
    440 1.091 4.355
    460 0.7701 3.362
    480 0.5474 2.612
    500 0.3916 2.042
    520 0.2819 1.605
    540 0.2042 1.267
    560 0.1488 1.005
    580 0.1092 0.7997
    600 0.08070 0.6390
    620 0.06012 0.5123
    640 0.04519 0.4121
    660 0.03430 0.3325
    680 0.02632 0.2691
    700 0.02043 0.2185
    720 0.01607 0.1779
    740 0.01281 0.1452
    760 0.01036 0.1190
    780 0.008496 0.09776
    800 0.007069 0.08059
    840 0.004680 0.05741
    880 0.003200 0.04210
    920 0.002210 0.03130
    960 0.001560 0.02360
    1000 0.001150 0.01810
]

const _HARRIS_PRIESTER_HEIGHTS = _HARRIS_PRIESTER_DATA[:, 1]
const _HARRIS_PRIESTER_RHO_MIN = _HARRIS_PRIESTER_DATA[:, 2]
const _HARRIS_PRIESTER_RHO_MAX = _HARRIS_PRIESTER_DATA[:, 3]
const _HARRIS_PRIESTER_ITP_MIN = interpolate(
    _HARRIS_PRIESTER_HEIGHTS,
    _HARRIS_PRIESTER_RHO_MIN,
    FiniteDifferenceMonotonicInterpolation(),
)
const _HARRIS_PRIESTER_ITP_MAX = interpolate(
    _HARRIS_PRIESTER_HEIGHTS,
    _HARRIS_PRIESTER_RHO_MAX,
    FiniteDifferenceMonotonicInterpolation(),
)
const _G_KM3_TO_KG_M3 = 1e-12


"""
    HarrisPriesterModel(h_km; use_min=true)

Interpolate the Harris-Priester atmospheric density look-up table.

# Arguments
- `h_km`: geodetic altitude above the reference radius, in km
- `use_min`: if `true`, use the low solar-activity column; otherwise use the high column

# Returns
Atmospheric density in kg/m³.

Table values are stored in g/km³ and converted via a factor of `1e-12`.
Cubic Hermite splines (`FiniteDifferenceMonotonicInterpolation`) are used on the
irregular height grid; segment lookup uses the primal altitude so the result is
ForwardDiff-compatible.
Altitude is clamped to the tabulated range [100, 1000] km.
"""
function HarrisPriesterModel(h_km::Real; use_min::Bool=true)
    h_lo, h_hi = _HARRIS_PRIESTER_HEIGHTS[1], _HARRIS_PRIESTER_HEIGHTS[end]
    h = min(max(h_km, h_lo), h_hi)
    itp = use_min ? _HARRIS_PRIESTER_ITP_MIN : _HARRIS_PRIESTER_ITP_MAX
    return itp(h) * _G_KM3_TO_KG_M3
end


"""
    harris_priester_f_density(R_earth_km=6378.0; use_min=true)

Build an `f_density` callback using the Harris-Priester model that returns the density in kg/m^3

# Arguments
- `R_earth_km`: reference Earth radius used to compute altitude from `norm(r_km)`, in km
- `use_min`: if `true`, use the low solar-activity column; otherwise use the high column

# Returns
A callable `(et, r_km) -> rho` returning atmospheric density in kg/m³.
`r_km` must be in the planet-centered planet-fixed frame (`frame_PCPF`) used by the
EOM when `include_drag` is enabled; the epoch argument is accepted for API compatibility
but is not used by this model.
"""
function harris_priester_f_density(R_earth_km::Real=6378.0; use_min::Bool=true)
    R_earth_km = float(R_earth_km)
    function f_density(et, r_km)
        h_km = norm(r_km) - R_earth_km
        return HarrisPriesterModel(h_km; use_min=use_min)
    end
    return f_density
end