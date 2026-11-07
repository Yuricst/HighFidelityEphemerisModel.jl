"""Tests for the SPICE-free analytic Sun, sidereal time and UTC helpers"""


if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "utils.jl"))
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end


function _angle_deg(a, b)
    return acosd(clamp(dot(a, b) / (norm(a) * norm(b)), -1.0, 1.0))
end


function test_builtin_utc()
    H = HighFidelityEphemerisModel
    for date in ("1950-01-01T00:00:00", "1971-12-31T23:59:00", "1975-03-01T00:00:00", "1999-06-01T00:00:00", "2008-12-31T23:59:00",
                 "2012-07-01T00:00:10", "2017-01-01T00:00:01", "2020-01-01T00:00:00",
                 "2026-05-05T12:00:00")
        et = str2et(date)
        # agreement with SPICE deltet (naif0012.tls) to well below a microsecond
        @test abs(H.et_minus_utc_builtin(et) - deltet(et, "ET")) < 1e-6
        @test abs(H.et_to_utc_mjd_builtin(et) - H.et_to_utc_mjd(et)) * 86400 < 1e-6
    end
end


function test_sun_position_analytic()
    H = HighFidelityEphemerisModel
    # Astronomical Almanac low-precision Sun: ~0.01 deg over 1950-2050
    for year in 1960:10:2050
        et = str2et("$(year)-06-01T00:00:00")
        r_spice, _ = spkpos("SUN", et, "J2000", "NONE", "EARTH")
        r = H.sun_position_analytic(et)
        @test _angle_deg(r, r_spice) < 0.015
        @test abs(norm(r) / norm(r_spice) - 1) < 2e-4
    end

    # Earth-fixed (GMST-rotated) Sun vs DE440 + ITRF93 (within the Earth PCK coverage)
    for date in ("2000-03-20T00:00:00", "2010-09-15T06:00:00", "2020-01-01T00:00:00",
                 "2023-03-03T18:00:00")
        et = str2et(date)
        r_itrf, _ = spkpos("SUN", et, "ITRF93", "NONE", "EARTH")
        r_iau, _ = spkpos("SUN", et, "IAU_EARTH", "NONE", "EARTH")
        r = H.sun_position_earth_fixed_analytic(et)
        @test _angle_deg(r, r_itrf) < 0.015
        @test _angle_deg(r, r_iau) < 0.12
    end

    # ForwardDiff in time: matches central differences, and the Earth-fixed Sun moves
    # westward at ≈ the Earth rotation rate minus the mean solar motion
    et = str2et("2020-01-01T00:00:00")
    f = H.sun_position_earth_fixed_analytic
    dr = ForwardDiff.derivative(f, et)
    @test dr ≈ (f(et + 1.0) - f(et - 1.0)) / 2 rtol=1e-6
    r = f(et)
    ω = -(r[1] * dr[2] - r[2] * dr[1]) / (r[1]^2 + r[2]^2)
    @test ω ≈ 7.2921159e-5 - 2π / (365.2422 * 86400) rtol=1e-3
end


test_builtin_utc()
test_sun_position_analytic()
