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
    spice_dir = ENV["SPICE"]

    return (
        lsk = joinpath(spice_dir, "lsk", "naif0012.tls"),
        spk = joinpath(spice_dir, "spk", "de440.bsp"),
        gm = joinpath(spice_dir, "pck", "gm_de440.tpc"),
        bpc = joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc"),
        fk = joinpath(spice_dir, "fk", "moon_de440_250416.tf"),
    )
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


function test_ephemerides_segment_fallbacks()
    paths = furnish_ephemerides_test_kernels()
    provider = Ephemerides.EphemerisProvider(paths.spk)
    et = str2et("2026-01-05T00:00:00")

    r_sun_emb = HighFidelityEphemerisModel.get_pos_ephemerides(provider, "10", "3", et)
    r_sun_emb_spice, _ = spkpos("10", et, "J2000", "NONE", "3")
    @test maximum(abs.(collect(r_sun_emb) .- r_sun_emb_spice)) < 1e-8

    r_earth_moon = HighFidelityEphemerisModel.get_pos_ephemerides(provider, "399", "301", et)
    r_earth_moon_spice, _ = spkpos("399", et, "J2000", "NONE", "301")
    @test maximum(abs.(collect(r_earth_moon) .- r_earth_moon_spice)) < 1e-8

    r_sun_moon = HighFidelityEphemerisModel.get_pos_ephemerides(provider, "10", "301", et)
    r_sun_moon_spice, _ = spkpos("10", et, "J2000", "NONE", "301")
    @test maximum(abs.(collect(r_sun_moon) .- r_sun_moon_spice)) < 1e-6

    x_earth_moon = HighFidelityEphemerisModel.get_state_ephemerides(provider, "399", "301", et)
    x_earth_moon_spice, _ = spkezr("399", et, "J2000", "NONE", "301")
    @test maximum(abs.(collect(x_earth_moon) .- x_earth_moon_spice)) < 1e-8
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

    @test maximum(abs.(dx_ephem .- dx_spice)) < 1e-9
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
