"""
Test generic EOM dispatch.
"""

using SPICE
using LinearAlgebra
using Test

if !@isdefined(furnsh_kernels)
    include(joinpath(@__DIR__, "utils.jl"))
end

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end


function test_eom_Nbody_dispatch_SPICE()
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        0.0,
        1.0,
        [1.0],
        ["399"],
    )
    x = [1.0, 0.1, 0.2, 0.0, 0.3, -0.1]

    dx = HighFidelityEphemerisModel.eom_Nbody(x, parameters, 0.0)
    dx_backend = HighFidelityEphemerisModel.eom_Nbody_SPICE(x, parameters, 0.0)
    dx_inplace = zeros(6)
    jac_fd_backend = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody_SPICE, x, 0.0, parameters, 0.0
    )
    jac_generic = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody, x, 0.0, parameters, 0.0
    )
    x_stm = [x; reshape(I(6), 36)]
    dx_stm_backend = zeros(42)
    dx_stm_fd = zeros(42)

    HighFidelityEphemerisModel.eom_Nbody!(dx_inplace, x, parameters, 0.0)
    HighFidelityEphemerisModel.eom_stm_Nbody_SPICE!(dx_stm_backend, x_stm, parameters, 0.0)
    HighFidelityEphemerisModel.eom_stm_Nbody_SPICE_fd!(dx_stm_fd, x_stm, parameters, 0.0)

    @test parameters isa HighFidelityEphemerisModel.SpiceParameters
    @test dx == dx_backend
    @test dx_inplace == dx_backend
    @test isapprox(jac_fd_backend, jac_generic, atol = 1e-14)
    @test dx_stm_backend == dx_stm_fd
    @test parameters.ephemerides_frame_system === nothing
end


function test_eom_Nbody_dispatch_Interp()
    et0 = str2et("2026-01-05T00:00:00")
    etf = et0 + 86400.0
    naif_ids = ["399", "301"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0,
        1e5,
        GMs,
        naif_ids;
        interpolate_ephem_span = [et0, etf],
        interpolation_time_step = 3600.0,
    )
    x = [1.0, 0.1, 0.2, 0.0, 0.3, -0.1]

    dx = HighFidelityEphemerisModel.eom_Nbody(x, parameters, 0.0)
    dx_backend = HighFidelityEphemerisModel.eom_Nbody_Interp(x, parameters, 0.0)
    dx_inplace = zeros(6)
    jac_fd_backend = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody_Interp, x, 0.0, parameters, 0.0
    )
    jac_generic = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody, x, 0.0, parameters, 0.0
    )
    x_stm = [x; reshape(I(6), 36)]
    dx_stm_backend = zeros(42)
    dx_stm_fd = zeros(42)

    HighFidelityEphemerisModel.eom_Nbody!(dx_inplace, x, parameters, 0.0)
    HighFidelityEphemerisModel.eom_stm_Nbody_Interp!(dx_stm_backend, x_stm, parameters, 0.0)
    HighFidelityEphemerisModel.eom_stm_Nbody_Interp_fd!(dx_stm_fd, x_stm, parameters, 0.0)

    @test parameters isa HighFidelityEphemerisModel.InterpParameters
    @test dx == dx_backend
    @test dx_inplace == dx_backend
    @test jac_fd_backend == jac_generic
    @test dx_stm_backend == dx_stm_fd
    @test parameters.ephemerides_provider === nothing
end


function test_srp_requires_sun()
    args = (0.0, 1.0, [1.0], ["399"])

    @test_throws ArgumentError HighFidelityEphemerisModel.SpiceParameters(
        args...; include_srp = true
    )
    @test_throws ArgumentError HighFidelityEphemerisModel.InterpParameters(
        args...;
        include_srp = true,
        interpolate_ephem_span = [0.0, 1.0],
    )
    @test_throws ArgumentError HighFidelityEphemerisModel.EphemeridesParameters(
        args...;
        include_srp = true,
        ephemerides_frame_system = :unused,
    )
    @test_throws ArgumentError HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        args...; include_srp = true
    )
end


test_eom_Nbody_dispatch_SPICE()
test_eom_Nbody_dispatch_Interp()
test_srp_requires_sun()