"""Test transformation interpolation"""

using LinearAlgebra
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


test_interpolate_transformation_moon_pa = function ()
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
        @test T_spice ≈ T_interp atol=1e-11
    end
end


"""
Regression: IAU_EARTH spin wraps through ±π under m2eul. Without unwrapping,
spline interpolation between knots returns nonsensical rotation matrices and
silently corrupts Interp drag / spherical-harmonics body-fixed accelerations.
"""
test_interpolate_transformation_iau_earth_unwraps = function ()
    et0 = str2et("2026-01-05T00:00:00")
    ets = collect(range(et0, et0 + 2 * 86400.0; step = 3600.0))

    # Demonstrate the wrapped raw angle history that previously broke splines.
    raw_angles = zeros(3, length(ets))
    for (idx, et) in enumerate(ets)
        raw_angles[:, idx] .= m2eul(SPICE.pxform("J2000", "IAU_EARTH", et), 3, 1, 3)
    end
    raw_jumps = count(i -> abs(raw_angles[1, i] - raw_angles[1, i - 1]) > π, 2:length(ets))
    @test raw_jumps >= 1

    transformation_interp = HighFidelityEphemerisModel.InterpolatedTransformation(
        ets,
        "J2000",
        "IAU_EARTH",
        false,
        1.0,
    )

    max_err = 0.0
    for i in 1:(length(ets) - 1)
        et = 0.5 * (ets[i] + ets[i + 1])
        T_spice = SPICE.pxform("J2000", "IAU_EARTH", et)
        T_interp = HighFidelityEphemerisModel.pxform(transformation_interp, et)
        max_err = max(max_err, maximum(abs.(T_spice .- T_interp)))
    end

    # Before the unwrap fix this exceeded O(1); after unwrap it stays near knot accuracy.
    @test max_err < 1e-10
end


test_unwrap_euler_angles_helper = function ()
    angles = [
        0.0  3.0  -3.0  -2.0;
        0.1  0.1   0.1   0.1;
        0.0  0.0   0.0   0.0;
    ]
    HighFidelityEphemerisModel.unwrap_euler_angles!(angles)
    @test angles[1, 3] ≈ -3.0 + 2π
    @test angles[1, 4] ≈ -2.0 + 2π
    @test angles[2, :] ≈ fill(0.1, 4)
end


test_unwrap_euler_angles_helper()
test_interpolate_transformation_iau_earth_unwraps()
test_interpolate_transformation_moon_pa()
