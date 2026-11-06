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


function test_eom_Nbody_dispatch_Ephemerides()
    et0 = str2et("2026-01-05T00:00:00")
    naif_ids = ["399", "301"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0,
        1e5,
        GMs,
        naif_ids;
        ephemerides_files = hfem_test_kernel("spk", "de440.bsp"),
    )
    x = [1.0, 0.1, 0.2, 0.0, 0.3, -0.1]

    dx = HighFidelityEphemerisModel.eom_Nbody(x, parameters, 0.0)
    dx_backend = HighFidelityEphemerisModel.eom_Nbody_Ephemerides(x, parameters, 0.0)
    dx_inplace = zeros(6)
    jac_fd_backend = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody_Ephemerides, x, 0.0, parameters, 0.0
    )
    jac_generic = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody, x, 0.0, parameters, 0.0
    )
    x_stm = [x; reshape(I(6), 36)]
    dx_stm_backend = zeros(42)
    dx_stm_fd = zeros(42)

    HighFidelityEphemerisModel.eom_Nbody!(dx_inplace, x, parameters, 0.0)
    HighFidelityEphemerisModel.eom_stm_Nbody_Ephemerides!(dx_stm_backend, x_stm, parameters, 0.0)
    HighFidelityEphemerisModel.eom_stm_Nbody_Ephemerides_fd!(dx_stm_fd, x_stm, parameters, 0.0)

    @test parameters isa HighFidelityEphemerisModel.EphemeridesParameters
    @test dx == dx_backend
    @test dx_inplace == dx_backend
    @test jac_fd_backend == jac_generic
    @test dx_stm_backend == dx_stm_fd
    @test parameters.ephemerides_provider isa HighFidelityEphemerisModel.Ephemerides.EphemerisProvider
end


function test_removed_interp_backend_inputs()
    # the interpolated backend was removed: its inputs must not be silently ignored
    et0 = str2et("2026-01-05T00:00:00")
    naif_ids = ["399", "301"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    args = (et0, 1e5, GMs, naif_ids, "J2000", "NONE")
    span = [et0, et0 + 86400.0]
    files = hfem_test_kernel("spk", "de440.bsp")
    for ctor in (HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters,
                 HighFidelityEphemerisModel.SpiceParameters)
        @test_throws ArgumentError ctor(args...; interpolate_ephem_span = span)
        @test_throws ArgumentError ctor(args...; interpolation_time_step = 60.0)
    end
    @test_throws ArgumentError HighFidelityEphemerisModel.EphemeridesParameters(args...;
        ephemerides_files = files, interpolate_ephem_span = span)
    @test_throws ArgumentError HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(args...;
        backend = :interp)
    err = try
        HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(args...; interpolate_ephem_span = span)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("EphemeridesParameters", err.msg) && occursin("removed", err.msg)
    @test !isdefined(HighFidelityEphemerisModel, :InterpParameters)
end


test_eom_Nbody_dispatch_SPICE()
test_eom_Nbody_dispatch_Ephemerides()
test_removed_interp_backend_inputs()