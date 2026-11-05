"""Test transformation interpolation"""

using LinearAlgebra
using SPICE

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


test_interpolate_ephem = function ()
    # define parameters
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 3000.0

    et0 = str2et("2020-01-01T00:00:00")
    parameters = HighFidelityEphemerisModel.SpiceParameters(et0, DU, GMs, naif_ids, naif_frame, abcorr)

    # query states to be interpolated
    ets = range(et0, et0 + 30 * 86400.0, 1000)

    # interpolated transformation struct
    transformation_interp = HighFidelityEphemerisModel.InterpolatedTransformation(
        ets,
        "J2000",
        "MOON_PA",
        false,
        parameters.TU,
    )

    # evaluate position
    ets_test = range(et0 + 1e-4, et0 + 30 * 86400.0 - 1e-4, 12)
    Ts_spice = [SPICE.pxform("J2000", "MOON_PA", et) for et in ets_test]

    Ts_interp = [HighFidelityEphemerisModel.pxform(transformation_interp, et) for et in ets_test]

    for (T_spice, T_interp) in zip(Ts_spice, Ts_interp)
        # println("SPICE T:")
        # print_matrix(T_spice)
        # println("Interpolated T:")
        # print_matrix(T_interp)
        # println()
        @test T_spice ≈ T_interp atol=1e-11
    end
end

test_interpolate_ephem()


# Rapidly rotating frame: the prime-meridian Euler angle wraps by 2π every day,
# which the spline must follow (regression test for angle unwrapping).
test_interpolate_transformation_wrapping = function ()
    et0 = str2et("2020-01-01T00:00:00")
    ets = range(et0, et0 + 3 * 86400.0, step = 3600.0)
    transformation_interp = HighFidelityEphemerisModel.InterpolatedTransformation(
        ets, "J2000", "IAU_EARTH", false, 1.0,
    )
    ets_test = range(et0 + 1e-3, et0 + 3 * 86400.0 - 1e-3, 2000)
    err_max = maximum(
        maximum(abs.(SPICE.pxform("J2000", "IAU_EARTH", et) -
                     HighFidelityEphemerisModel.pxform(transformation_interp, et)))
        for et in ets_test
    )
    @test err_max < 1e-10

    angles = [0.1, 3.0, -3.0, -0.5, 2.9, -3.1]
    HighFidelityEphemerisModel._unwrap_angles!(angles)
    @test all(abs.(diff(angles)) .< π)
    @test angles[3] ≈ -3.0 + 2π
end

test_interpolate_transformation_wrapping()