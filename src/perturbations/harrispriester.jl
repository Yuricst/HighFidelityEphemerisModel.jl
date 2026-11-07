"""Harris-Priester atmospheric density model (Montenbruck & Gill, Satellite Orbits, §3.5.2)"""


"""
Harris-Priester density table for mean solar activity (Montenbruck & Gill, *Satellite
Orbits* (2000), §3.5.2).

Columns are:
   h [km], rho_min [g/km^3], rho_max [g/km^3]

`rho_min` is the density at the antapex of the diurnal bulge (night-side minimum) and
`rho_max` the density at its apex (day-side maximum); both are for mean solar activity
(they are *not* low/high solar-activity columns).

Conversions of density: 1 g/km^3 = 1e-12 kg/m^3.
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

const _HARRIS_PRIESTER_HEIGHTS = Float64.(_HARRIS_PRIESTER_DATA[:, 1])
const _HARRIS_PRIESTER_RHO_MIN = Float64.(_HARRIS_PRIESTER_DATA[:, 2])
const _HARRIS_PRIESTER_RHO_MAX = Float64.(_HARRIS_PRIESTER_DATA[:, 3])

# Density scale heights of each table interval [h_i, h_{i+1}], in km:
#   H_i = (h_i - h_{i+1}) / ln(rho_{i+1} / rho_i)
const _HARRIS_PRIESTER_SCALE_MIN = [
    (_HARRIS_PRIESTER_HEIGHTS[i] - _HARRIS_PRIESTER_HEIGHTS[i+1]) /
    log(_HARRIS_PRIESTER_RHO_MIN[i+1] / _HARRIS_PRIESTER_RHO_MIN[i])
    for i in 1:length(_HARRIS_PRIESTER_HEIGHTS)-1
]
const _HARRIS_PRIESTER_SCALE_MAX = [
    (_HARRIS_PRIESTER_HEIGHTS[i] - _HARRIS_PRIESTER_HEIGHTS[i+1]) /
    log(_HARRIS_PRIESTER_RHO_MAX[i+1] / _HARRIS_PRIESTER_RHO_MAX[i])
    for i in 1:length(_HARRIS_PRIESTER_HEIGHTS)-1
]

const _HARRIS_PRIESTER_H_LOWER_KM = 100.0
const _HARRIS_PRIESTER_H_UPPER_KM = 1000.0
const _G_KM3_TO_KG_M3 = 1e-12

# WGS-84 ellipsoid (default reference ellipsoid for the geodetic altitude)
const _WGS84_RE_KM = 6378.137
const _WGS84_FLATTENING = 1 / 298.257223563


"""
    _harris_priester_index(h_km)

Index `i` of the table interval `[h_i, h_{i+1})` containing `h_km`, clamped to the first
and last intervals. Uses only comparisons, so it works for ForwardDiff duals (the
interval is selected by the primal value).
"""
@inline function _harris_priester_index(h_km::Real)
    H = _HARRIS_PRIESTER_HEIGHTS
    N = length(H)
    h_km < H[2] && return 1
    h_km >= H[N-1] && return N - 1
    lo, hi = 2, N - 1           # invariant: H[lo] <= h_km < H[hi]
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        if h_km < H[mid]
            hi = mid
        else
            lo = mid
        end
    end
    return lo
end


@inline function _harris_priester_column(h_km::Real, i::Int, use_min::Bool)
    if use_min
        return _HARRIS_PRIESTER_RHO_MIN[i] *
               exp((_HARRIS_PRIESTER_HEIGHTS[i] - h_km) / _HARRIS_PRIESTER_SCALE_MIN[i])
    else
        return _HARRIS_PRIESTER_RHO_MAX[i] *
               exp((_HARRIS_PRIESTER_HEIGHTS[i] - h_km) / _HARRIS_PRIESTER_SCALE_MAX[i])
    end
end


@inline _harris_priester_in_range(h_km::Real) =
    _HARRIS_PRIESTER_H_LOWER_KM < h_km < _HARRIS_PRIESTER_H_UPPER_KM


"""
    HarrisPriesterModel(h_km; use_min=true)

Density of a single Harris-Priester table column, in kg/m³ (no diurnal bulge).

# Arguments
- `h_km`: altitude above the reference ellipsoid, in km
- `use_min`: if `true`, use the bulge-antapex column `rho_min`; otherwise the bulge-apex
  column `rho_max` (both for mean solar activity)

The table is interpolated exponentially, with a constant density scale height in each
table interval (Montenbruck & Gill, *Satellite Orbits*, §3.5.2); the result
is exact at the tabulated heights and ForwardDiff-compatible. For backward
compatibility this helper clamps `h_km` to the tabulated range [100, 1000] km; the full
model [`harris_priester_density`](@ref) and the [`harris_priester_f_density`](@ref)
callback instead return zero outside that range, as in M&G.

!!! note "Changed after v0.2.3"
    Previously the column was interpolated with a monotone cubic spline in ρ. It now uses
    M&G's piecewise-exponential interpolation; values at table heights are unchanged.
"""
function HarrisPriesterModel(h_km::Real; use_min::Bool = true)
    h = min(max(h_km, _HARRIS_PRIESTER_H_LOWER_KM), _HARRIS_PRIESTER_H_UPPER_KM)
    return _harris_priester_column(h, _harris_priester_index(h), use_min) * _G_KM3_TO_KG_M3
end


"""
    harris_priester_density(h_km, cos_psi; n=6)

Harris-Priester atmospheric density with diurnal bulge, in kg/m³ (Montenbruck & Gill,
*Satellite Orbits* (2000), §3.5.2):

    ρ(h, ψ) = ρ_min(h) + (ρ_max(h) - ρ_min(h)) cos^n(ψ/2)

where `ρ_min`/`ρ_max` are the antapex/apex table densities interpolated exponentially in
height, and `ψ` is the angle between the satellite position vector and the apex of the
diurnal bulge, passed here as `cos_psi = cos ψ` (`cos²(ψ/2) = (1 + cos ψ)/2`).

# Arguments
- `h_km`: altitude above the reference ellipsoid, in km
- `cos_psi`: cosine of the angle between the satellite position and the bulge apex
- `n`: exponent of `cos(ψ/2)`. M&G recommend `n = 2` for low-inclination orbits and
  `n = 6` for polar orbits; intermediate values may be used for intermediate inclinations.

Returns zero outside the model's validity range 100 km < h < 1000 km (as M&G's reference
implementation). ForwardDiff-compatible in `h_km` and `cos_psi`.

!!! warning "Only C⁰ in altitude"
    The piecewise-exponential profile is continuous but `∂ρ/∂h` jumps at every table
    height (the scale height changes from one interval to the next), and `ρ` jumps to zero
    at 100 km and 1000 km. See [`harris_priester_f_density`](@ref) for the consequences
    for STMs and optimization.
"""
function harris_priester_density(h_km::Real, cos_psi::Real; n::Real = 6)
    n >= 0 || throw(ArgumentError("Harris-Priester exponent `n` must be non-negative, got $n"))
    T = float(promote_type(typeof(h_km), typeof(cos_psi)))
    _harris_priester_in_range(h_km) || return zero(T)
    i = _harris_priester_index(h_km)
    d_min = _harris_priester_column(h_km, i, true)
    d_max = _harris_priester_column(h_km, i, false)
    c2 = clamp((1 + cos_psi) / 2, zero(cos_psi), one(cos_psi))   # cos²(ψ/2), guard roundoff
    fac = (isinteger(n) && iseven(Int(n))) ? c2^(Int(n) ÷ 2) : c2^(n / 2)
    return (d_min + (d_max - d_min) * fac) * _G_KM3_TO_KG_M3
end


"""
    harris_priester_f_density(R_earth_km=6378.137;
        flattening=1/298.257223563, n=6, lag_deg=30.0,
        sun_position=nothing, use_min=nothing)

Build an `f_density(et, r_km)` callback for the Harris-Priester model of Montenbruck &
Gill, *Satellite Orbits* (2000), §3.5.2, returning the density in kg/m³.

`r_km` is the spacecraft position in km in the Earth body-fixed frame `frame_PCPF`
(e.g. `"ITRF93"` or `"IAU_EARTH"`), which is what the drag EOMs pass to `f_density`;
`et` is the epoch (TDB seconds past J2000).

!!! warning "`r_km` must be Earth-fixed"
    With the default analytic Sun, the Sun (hence the bulge apex) is computed in an
    Earth-fixed frame, so the density is only correct if `r_km` is Earth-fixed too, i.e.
    if `frame_PCPF` is an Earth body-fixed frame such as `"ITRF93"` or `"IAU_EARTH"`.
    Passing an inertial position (e.g. `frame_PCPF = "J2000"`, or calling the callback
    with J2000 coordinates) silently places the bulge at the wrong local time (e.g. a
    factor ≈0.6 in density at 400 km for one test case). Unlike the previous
    altitude-only model, the result is not invariant under rotations of `r_km`. If you
    supply `sun_position`, it must return the Sun in the same frame as `r_km`.

!!! warning "Zero density below 100 km and above 1000 km"
    Following M&G, the density is **zero** outside 100 km < h < 1000 km. In particular a
    re-entering spacecraft loses all drag below 100 km (geodetic), and the density jumps
    from ≈5e-7 kg/m³ to 0 there, which an adaptive integrator will see as a discontinuity.
    Stop the propagation with an altitude event (e.g. a `ContinuousCallback` at 100 km)
    or use a model valid below 100 km. (`jacchia_roberts_f_density` throws an error at
    or below 100 km instead.)

!!! warning "Only C⁰ in altitude: STMs and gradients"
    `∂ρ/∂h` jumps at every table height (10 km spacing below 300 km, 20-40 km above), so
    the drag acceleration is not differentiable there. Variational equations / STMs
    computed with ForwardDiff remain the derivative of the model almost everywhere, but
    on an orbit that crosses table heights they lose accuracy against finite-difference
    STMs (discrepancies of ~4e-4 were observed over a few hours, vs ≲1e-5 on an orbit
    that stays inside one interval), and adaptive integrators step across the kinks
    unaware. Gradient-based optimization (e.g. of drag-dependent trajectories) may see
    non-smooth objectives. Prefer a smooth density model if this matters.

The model:
1. altitude `h` is the geodetic height above an oblate ellipsoid with equatorial radius
   `R_earth_km` and `flattening` (pass `flattening=0` for a spherical altitude
   `norm(r_km) - R_earth_km`);
2. the antapex/apex densities `ρ_min(h)`, `ρ_max(h)` are interpolated exponentially
   between table heights;
3. the apex of the diurnal bulge lies at the Sun's declination and `lag_deg` (≈30°,
   i.e. ~14h local time) east of the Sun in right ascension / longitude;
4. `ρ = ρ_min + (ρ_max - ρ_min) cos^n(ψ/2)`, `ψ` being the angle between `r_km` and the
   apex (see [`harris_priester_density`](@ref)); zero outside 100 km < h < 1000 km.

# Arguments
- `R_earth_km`: equatorial radius of the reference ellipsoid, in km (default WGS-84)
- `flattening`: ellipsoid flattening (default WGS-84)
- `n`: bulge exponent; M&G recommend 2 for low-inclination and 6 for polar orbits
- `lag_deg`: lag of the bulge apex behind the Sun in right ascension, in degrees
- `sun_position`: `nothing` (default) to use the SPICE-free low-precision analytic Sun
  ([`sun_position_earth_fixed_analytic`](@ref), accurate to ~0.01° w.r.t. ITRF93; w.r.t.
  IAU_EARTH the difference grows from ≈0.05° in 2000 to 0.1° in 2030, 0.15° in 2050 and
  0.27° in 2100 because IAU_EARTH drifts from the true Earth rotation; all negligible for
  this model), or a
  function `et -> r_sun` returning the Sun position (any length unit) in the **same
  frame as `r_km`**, e.g. one built with [`ephemerides_sun_position_function`](@ref).
  A SPICE-based function such as
  `et -> spkpos("SUN", et, "ITRF93", "NONE", "EARTH")[1]` also works but makes the
  callback non-thread-safe.
- `use_min`: legacy option. `nothing` (default) evaluates the full model with the
  diurnal bulge. `true`/`false` instead return only the `ρ_min`/`ρ_max` column (no
  bulge, no Sun position needed), still with geodetic altitude, exponential
  interpolation, and zero density outside 100-1000 km.

The callback is SPICE-free (unless `sun_position` calls SPICE), allocation-light,
thread-safe, and ForwardDiff-compatible in `r_km` (and `et`).

!!! note "Changed after v0.2.3"
    Earlier versions used the spherical altitude `norm(r_km) - R_earth_km` with
    `R_earth_km = 6378.0` by default, a cubic spline in ρ, clamped the altitude to
    [100, 1000] km, and had no diurnal bulge (`use_min=true` selected the `ρ_min` column
    by default). The default is now the full M&G model above. To approximate the old
    behaviour, use `harris_priester_f_density(6378.0; flattening=0, use_min=true)`.
"""
function harris_priester_f_density(
    R_earth_km::Real = _WGS84_RE_KM;
    flattening::Real = _WGS84_FLATTENING,
    n::Real = 6,
    lag_deg::Real = 30.0,
    sun_position = nothing,
    use_min::Union{Nothing,Bool} = nothing,
)
    n >= 0 || throw(ArgumentError("Harris-Priester exponent `n` must be non-negative, got $n"))
    Re = float(R_earth_km)
    flat = float(flattening)

    if !isnothing(use_min)
        column_min = use_min
        function f_density_column(et, r_km)
            _, _, h_km = _geodetic_lon_lat_alt(r_km, Re, flat)
            T = float(typeof(h_km))
            _harris_priester_in_range(h_km) || return zero(T)
            i = _harris_priester_index(h_km)
            return _harris_priester_column(h_km, i, column_min) * _G_KM3_TO_KG_M3
        end
        return f_density_column
    end

    slag, clag = sincos(deg2rad(float(lag_deg)))
    n_exp = n
    get_sun = isnothing(sun_position) ? _sun_position_earth_fixed_analytic : sun_position

    function f_density(et, r_km)
        _, _, h_km = _geodetic_lon_lat_alt(r_km, Re, flat)
        _harris_priester_in_range(h_km) || return zero(float(typeof(h_km)))
        s = get_sun(et)
        # unit vector of the bulge apex: Sun direction rotated by +lag about the pole
        ux = clag * s[1] - slag * s[2]
        uy = slag * s[1] + clag * s[2]
        uz = s[3]
        x, y, z = r_km[1], r_km[2], r_km[3]
        cos_psi = (x * ux + y * uy + z * uz) /
                  (sqrt(x^2 + y^2 + z^2) * sqrt(ux^2 + uy^2 + uz^2))
        return harris_priester_density(h_km, cos_psi; n = n_exp)
    end
    return f_density
end
