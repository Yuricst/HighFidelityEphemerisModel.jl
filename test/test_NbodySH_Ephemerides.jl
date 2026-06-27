"""
Test eom_NbodySH_Ephemerides and Ephemerides.jl frame/segment fallback helpers.
"""

using Ephemerides
using LinearAlgebra
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


function ephemerides_test_kernel_paths()
    if haskey(ENV, "SPICE")
        spice_dir = ENV["SPICE"]

        return (
            lsk = joinpath(spice_dir, "lsk", "naif0012.tls"),
            spk = joinpath(spice_dir, "spk", "de440.bsp"),
            gm = joinpath(spice_dir, "pck", "gm_de440.tpc"),
            bpc = joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc"),
            fk = joinpath(spice_dir, "fk", "moon_de440_250416.tf"),
        )
    else
        spice_dir = joinpath(@__DIR__, "../spice/test")

        return (
            lsk = joinpath(spice_dir, "naif0012.tls"),
            spk = joinpath(spice_dir, "de440.bsp"),
            gm = joinpath(spice_dir, "gm_de440.tpc"),
            bpc = joinpath(spice_dir, "moon_pa_de440_200625.bpc"),
            fk = joinpath(spice_dir, "moon_de440_250416.tf"),
        )
    end
end


function furnish_ephemerides_test_kernels()
    paths = ephemerides_test_kernel_paths()

    for path in values(paths)
        isfile(path) || error("Required SPICE kernel does not exist: $(path)")
        furnsh(path)
    end

    return paths
end


function constant_density(et, r_km)
    return 1e-12
end


function moon_spherical_harmonics_path()
    package_root = pkgdir(HighFidelityEphemerisModel)

    if isnothing(package_root)
        package_root = normpath(joinpath(@__DIR__, ".."))
    end

    return joinpath(
        package_root,
        "data",
        "luna",
        "gggrx_1200l_sha_20x20.tab",
    )
end


function test_pxform_ephemerides()
    paths = furnish_ephemerides_test_kernels()

    et0 = str2et("2026-01-05T00:00:00")
    params = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0,
        3000.0,
        [1.0, 1.0e-6],
        ["3", "399"],
        "J2000",
        "NONE";
        frame_PCPF = "MOON_PA",
        ephemerides_files = [paths.spk, paths.bpc],
    )

    T_ephem = HighFidelityEphemerisModel.pxform_ephemerides(
        params,
        "J2000",
        "MOON_PA",
        et0,
    )
    T_spice = pxform("J2000", "MOON_PA", et0)

    @test maximum(abs.(T_ephem .- T_spice)) < 1e-10
end


function _test_ephemerides_position_close(frames, target, center, et; atol = 1e-8, rtol = 5e-16)
    r_ephem = HighFidelityEphemerisModel.get_pos_ephemerides(
        frames,
        target,
        center,
        et;
        axes = "J2000",
    )
    r_spice, _ = spkpos(string(target), et, "J2000", "NONE", string(center))

    @test isapprox(collect(r_ephem), r_spice; atol = atol, rtol = rtol)
end


function _test_ephemerides_state_close(frames, target, center, et; pos_atol = 1e-8, vel_atol = 5e-14, rtol = 5e-16)
    x_ephem = collect(
        HighFidelityEphemerisModel.get_state_ephemerides(
            frames,
            target,
            center,
            et;
            axes = "J2000",
        )
    )
    x_spice, _ = spkezr(string(target), et, "J2000", "NONE", string(center))

    @test isapprox(x_ephem[1:3], x_spice[1:3]; atol = pos_atol, rtol = rtol)
    @test isapprox(x_ephem[4:6], x_spice[4:6]; atol = vel_atol, rtol = rtol)
end


function test_ephemerides_segment_fallbacks()
    paths = furnish_ephemerides_test_kernels()

    # Build the documented Ephemerides + FrameTransformations path.  Ephemerides.jl
    # only reads stored SPK/PCK records; FrameTransformations.jl performs the
    # point-chain concatenation for arbitrary target/center pairs.
    provider = Ephemerides.EphemerisProvider(paths.spk)
    frames = HighFidelityEphemerisModel.build_ephemerides_frame_system(provider)
    et = str2et("2026-01-05T00:00:00")

    # Stored or near-direct common cases should be much tighter than the old
    # blanket 1e-6 km fallback tolerance.
    _test_ephemerides_position_close(frames, "10", "3", et; atol = 5e-8, rtol = 5e-16)
    _test_ephemerides_position_close(frames, "399", "301", et; atol = 1e-8, rtol = 5e-16)
    _test_ephemerides_position_close(frames, "10", "301", et; atol = 5e-8, rtol = 5e-16)

    _test_ephemerides_state_close(frames, "399", "301", et; pos_atol = 1e-8, vel_atol = 5e-14)
    _test_ephemerides_state_close(frames, "10", "301", et; pos_atol = 5e-8, vel_atol = 5e-14)

    # Planet-center IDs should chain through their planetary barycenters when
    # direct SPK segments are unavailable to Ephemerides.jl.
    for target in ("199", "299")
        _test_ephemerides_position_close(frames, target, "301", et; atol = 5e-8, rtol = 5e-16)
        _test_ephemerides_state_close(frames, target, "301", et; pos_atol = 5e-8, vel_atol = 5e-14)
    end

    # Outer barycenter wrt Moon cases involve billion-km vectors composed through
    # the point graph.  The remaining absolute differences vs CSPICE are near
    # Float64 roundoff for the vector scale, so use a tight relative tolerance
    # instead of a blanket absolute 1e-6 km tolerance.
    for target in ("4", "5", "6", "7", "8", "9")
        _test_ephemerides_position_close(frames, target, "301", et; atol = 1e-8, rtol = 5e-16)
        _test_ephemerides_state_close(frames, target, "301", et; pos_atol = 1e-8, vel_atol = 5e-14)
    end
end

function moon_centered_parameters(; backend::Symbol)
    paths = furnish_ephemerides_test_kernels()

    et0 = str2et("2026-01-05T00:00:00")
    naif_ids = ["301", "399", "10"]
    GMs = [
        bodvrd("MOON", "GM", 1)[1],
        bodvrd("EARTH", "GM", 1)[1],
        bodvrd("SUN", "GM", 1)[1],
    ]
    DU = 1e5

    kwargs = (
        filepath_spherical_harmonics = moon_spherical_harmonics_path(),
        nmax = 4,
        frame_PCPF = "MOON_PA",
        include_srp = true,
        include_drag = true,
        f_density = constant_density,
    )

    if backend == :spice
        return HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
            et0,
            DU,
            GMs,
            naif_ids,
            "J2000",
            "NONE";
            kwargs...,
        )
    elseif backend == :ephemerides
        return HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
            et0,
            DU,
            GMs,
            naif_ids,
            "J2000",
            "NONE";
            kwargs...,
            ephemerides_files = [paths.spk, paths.bpc],
        )
    else
        error("Unknown backend: $(backend)")
    end
end


function test_eom_NbodySH_Ephemerides_matches_SPICE()
    params_spice = moon_centered_parameters(backend = :spice)
    params_ephem = moon_centered_parameters(backend = :ephemerides)

    x0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]
    t = 0.1

    dx_spice = HighFidelityEphemerisModel.eom_NbodySH_SPICE(x0, params_spice, t)
    dx_ephem = HighFidelityEphemerisModel.eom_NbodySH_Ephemerides(x0, params_ephem, t)

    @test maximum(abs.(dx_ephem .- dx_spice)) < 1e-12
end


function test_eom_stm_NbodySH_Ephemerides_fd()
    params_ephem = moon_centered_parameters(backend = :ephemerides)

    x0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]
    x0_stm = [x0; reshape(I(6), 36)]
    t = 0.1

    dx_stm = zeros(42)
    HighFidelityEphemerisModel.eom_stm_NbodySH_Ephemerides_fd!(
        dx_stm,
        x0_stm,
        params_ephem,
        t,
    )

    dx = HighFidelityEphemerisModel.eom_NbodySH_Ephemerides(x0, params_ephem, t)

    @test all(isfinite, dx_stm)
    @test maximum(abs.(dx_stm[1:6] .- dx)) < 1e-12
    @test norm(dx_stm[7:42]) > 0.0
end


@testset "Ephemerides frame transforms" begin
    test_pxform_ephemerides()
end

@testset "Ephemerides segment fallbacks" begin
    test_ephemerides_segment_fallbacks()
end

@testset "NbodySH Ephemerides EOM" begin
    test_eom_NbodySH_Ephemerides_matches_SPICE()
    test_eom_stm_NbodySH_Ephemerides_fd()
end