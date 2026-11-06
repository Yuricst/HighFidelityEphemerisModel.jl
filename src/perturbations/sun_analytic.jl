"""SPICE-free, thread-safe helpers used by the atmosphere models: built-in leap-second
table, low-precision analytic Sun, and Greenwich mean sidereal time."""


# TAI - UTC [s] and the UTC modified Julian date from which each value applies
# (IERS Bulletin C; unchanged since 2017-01-01).
const _LEAP_SECOND_MJD = (
    41317.0, 41499.0, 41683.0, 42048.0, 42413.0, 42778.0, 43144.0, 43509.0, 43874.0,
    44239.0, 44786.0, 45151.0, 45516.0, 46247.0, 47161.0, 47892.0, 48257.0, 48804.0,
    49169.0, 49534.0, 50083.0, 50630.0, 51179.0, 53736.0, 54832.0, 56109.0, 57204.0,
    57754.0,
)
const _LEAP_SECOND_TAI_UTC = (
    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
    24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
)
const _MJD_J2000 = 51544.5


"""
    et_minus_utc_builtin(et)

ET (TDB) minus UTC, in seconds, from a built-in leap-second table (no SPICE call).

Mirrors SPICE `deltet(et, "ET")`: `ΔAT + 32.184 + K sin(E)` with the same periodic
TDB-TT term (K = 1.657e-3 s). The leap-second table covers 1972-01-01 to the
2017-01-01 leap second (TAI-UTC = 37 s). Epochs before 1972 use TAI-UTC = 9 s, as SPICE
`deltet` does with `naif0012.tls` (UTC was not defined with integer leap seconds before
1972, so neither value is the historical UTC). If IERS announces
a new leap second, results after it are off by 1 s (≈0.004° of Earth rotation).
Thread-safe and ForwardDiff-compatible in `et`.
"""
function et_minus_utc_builtin(et::Real)
    # periodic TDB - TT term, identical to SPICE's DELTET (naif0012.tls constants)
    M = 6.239996 + 1.99096871e-7 * et
    E = M + 1.671e-2 * sin(M)
    tdb_tt = 1.657e-3 * sin(E)
    tai_utc = _LEAP_SECOND_TAI_UTC[1] - 1.0     # before 1972: 9 s, as SPICE deltet
    for i in length(_LEAP_SECOND_MJD):-1:1
        # UTC seconds past J2000 at which the i-th leap-second value starts
        utc_start = (_LEAP_SECOND_MJD[i] - _MJD_J2000) * 86400.0
        if et - (32.184 + _LEAP_SECOND_TAI_UTC[i]) - tdb_tt >= utc_start
            tai_utc = _LEAP_SECOND_TAI_UTC[i]
            break
        end
    end
    return tai_utc + 32.184 + tdb_tt
end


"""
    et_to_utc_mjd_builtin(et)

SPICE-free counterpart of [`et_to_utc_mjd`](@ref): UTC modified Julian date from
ephemeris time (TDB seconds past J2000), using [`et_minus_utc_builtin`](@ref).
Thread-safe.
"""
et_to_utc_mjd_builtin(et::Real) = _MJD_J2000 + (et - et_minus_utc_builtin(et)) / 86400.0


"""
    sun_position_analytic(et; equinox_of_date=false)

Low-precision geocentric position of the Sun, in km, from the low-precision solar
formulas of *The Astronomical Almanac* (section C; also Vallado, *Fundamentals of
Astrodynamics and Applications*, Algorithm 29), quoted accurate to 0.01° for 1950-2050:

    n = days of TT past J2000
    L = 280.460° + 0.9856474° n                 (mean longitude, mean equinox of date)
    g = 357.528° + 0.9856003° n                 (mean anomaly)
    λ = L + 1.915° sin g + 0.020° sin 2g        (ecliptic longitude; β = 0)
    ε = 23.439° - 0.0000004° n                  (mean obliquity of date)
    r = 1.00014 - 0.01671 cos g - 0.00014 cos 2g   [AU]

(TDB ≈ TT is used for `et`). With `equinox_of_date=true` the result is in mean-of-date
equatorial axes. Otherwise (default) the longitude is referred to the J2000 equinox by
subtracting the general precession in longitude (1.3970° per Julian century) and the J2000
obliquity 23.43929111° is used, giving the Sun in EME2000 (≈ J2000/ICRF at this accuracy).
The simpler series of Montenbruck & Gill §3.3.2 (fixed longitude of perihelion) was not
used: compared with DE440 it drifts by ~0.003°/yr away from J2000 (0.08° in 2026).

Accuracy is ~0.01° in direction and ~1e-4 relative in distance, adequate for atmospheric
density models (diurnal bulge, local solar time, solar declination), not for SRP or
third-body gravity. Thread-safe; no SPICE call. Returns a 3-element `Vector`.
"""
function sun_position_analytic(et::Real; equinox_of_date::Bool = false)
    x, y, z = _sun_position_analytic(et, equinox_of_date)
    return [x, y, z]
end


const _AU_KM = 1.495978707e8

function _sun_position_analytic(et::Real, equinox_of_date::Bool)
    n = et / 86400.0
    L = 280.460 + 0.9856474 * n
    g = deg2rad(357.528 + 0.9856003 * n)
    λ = deg2rad(L + 1.915 * sin(g) + 0.020 * sin(2g))
    r = (1.00014 - 0.01671 * cos(g) - 0.00014 * cos(2g)) * _AU_KM
    if equinox_of_date
        ε = deg2rad(23.439 - 0.0000004 * n)
    else
        λ -= deg2rad(1.3970 * n / 36525.0)
        ε = deg2rad(23.43929111)
    end
    sλ, cλ = sincos(λ)
    sε, cε = sincos(ε)
    return r * cλ, r * sλ * cε, r * sλ * sε
end


"""
    gmst_builtin(et)

Greenwich mean sidereal time, in radians, from the IAU 1982 expression in the form
given by Meeus, *Astronomical Algorithms* (1998), Eq. (12.4)

    θ = 280.46061837° + 360.98564736629° d + 0.000387933° T² - T³/38710000

with `d` the UT1 days past J2000 and `T = d/36525`. UT1 is approximated by UTC
(|UT1-UTC| < 0.9 s, i.e. < 0.004°), obtained from the built-in leap-second table.
Thread-safe; no SPICE call.
"""
function gmst_builtin(et::Real)
    d = (et - et_minus_utc_builtin(et)) / 86400.0
    T = d / 36525.0
    θ = 280.46061837 + 360.98564736629 * d + T^2 * (0.000387933 - T / 38710000.0)
    return deg2rad(mod(θ, 360.0))
end


"""
    sun_position_earth_fixed_analytic(et)

Low-precision Sun position (km) in an Earth-fixed frame, without SPICE.

The mean-of-date analytic Sun ([`sun_position_analytic`](@ref) with
`equinox_of_date=true`) is rotated about the pole by the Greenwich mean sidereal time
([`gmst_builtin`](@ref)). Nutation (≤ ~20"), polar motion (≤ ~0.5") and UT1-UTC
(< 0.004°) are neglected, so the result approximates the Sun in `ITRF93` to roughly the
accuracy of the analytic series (≤ 0.011° measured against DE440 + ITRF93 for
2000-2026). `IAU_EARTH` (IAU 2009 rotation model) itself drifts away from the true Earth
rotation, so the difference to the Sun in `IAU_EARTH` grows with time: measured maxima
are ≈0.05° (2000), 0.08° (2020), 0.10° (2026-2030), 0.15° (2050), 0.21° (2075) and
0.27° (2100) (≈0.06° in 1960-1980). Both are far below the resolution of the atmosphere models that use this
function (Harris-Priester, Jacchia-Roberts). For exact consistency with a given `frame_PCPF`, supply your own Sun
position function to the density-model builders instead.

Thread-safe and ForwardDiff-compatible in `et`. Returns a 3-element `Vector`.
"""
function sun_position_earth_fixed_analytic(et::Real)
    x, y, z = _sun_position_earth_fixed_analytic(et)
    return [x, y, z]
end


function _sun_position_earth_fixed_analytic(et::Real)
    x, y, z = _sun_position_analytic(et, true)
    sθ, cθ = sincos(gmst_builtin(et))
    return cθ * x + sθ * y, -sθ * x + cθ * y, z
end


"""
    ephemerides_sun_position_function(frames, frame_PCPF; center_id=399)

Build a SPICE-free `sun_position(et)` callback, returning the Sun position (km) relative
to `center_id` in `frame_PCPF`, from a `FrameTransformations.FrameSystem` (e.g. the one
built by [`build_ephemerides_frame_system`](@ref) or `params.ephemerides_frame_system`
of an [`EphemeridesParameters`](@ref)). The frame system must contain the Sun (10),
`center_id`, and the `frame_PCPF` axes.

The result can be passed as `sun_position` to [`harris_priester_f_density`](@ref) or
[`jacchia_roberts_f_density`](@ref) when a DE-accurate Sun in exactly the propagation's
body-fixed frame is preferred over the analytic Sun. Thread safety is that of the
Ephemerides.jl provider (see [`ensure_ephemerides_thread_caches!`](@ref)).
"""
function ephemerides_sun_position_function(frames, frame_PCPF::String; center_id::Integer = 399)
    axes = ephemerides_axes_symbol(frame_PCPF)
    cid = Int(center_id)
    return et -> collect(FrameTransformations.vector3(frames, cid, 10, axes, et))
end
