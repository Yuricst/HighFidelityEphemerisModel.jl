"""
Test Earth body-fixed axes (ITRF93, IAU_EARTH) in the Ephemerides.jl/FrameTransformations
backend, and Earth-centred drag + spherical harmonics against the SPICE backend.
"""

using Ephemerides
using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "utils.jl"))
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end


function earth_ephemerides_kernel_paths()
    if haskey(ENV, "SPICE")
        spice_dir = ENV["SPICE"]
        return (
            spk = joinpath(spice_dir, "spk", "de440.bsp"),
            earth_bpc = joinpath(spice_dir, "pck", "earth_latest_high_prec.bpc"),
            moon_bpc = joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc"),
        )
    else
        spice_dir = joinpath(@__DIR__, "../spice/test")
        return (
            spk = joinpath(spice_dir, "de440.bsp"),
            earth_bpc = joinpath(spice_dir, "earth_latest_high_prec.bpc"),
            moon_bpc = joinpath(spice_dir, "moon_pa_de440_200625.bpc"),
        )
    end
end


function _earth_parameters(backend::Symbol, frame_PCPF::String; et0, nmax = 8,
                           include_drag = true, f_density = nothing)
    paths = earth_ephemerides_kernel_paths()
    naif_ids = ["399", "301", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    DU = 6378.0
    f_density = isnothing(f_density) ? HighFidelityEphemerisModel.harris_priester_f_density() : f_density
    kwargs = (
        filepath_spherical_harmonics = joinpath(@__DIR__, "../data/earth/GGM03S.tab"),
        nmax = nmax,
        frame_PCPF = frame_PCPF,
        include_srp = true,
        include_drag = include_drag,
        drag_Cd = 2.2,
        drag_Am = 0.01,
        f_density = include_drag ? f_density : nothing,
    )
    if backend == :spice
        return HighFidelityEphemerisModel.SpiceParameters(
            et0, DU, GMs, naif_ids, "J2000", "NONE"; kwargs...,
        )
    else
        return HighFidelityEphemerisModel.EphemeridesParameters(
            et0, DU, GMs, naif_ids, "J2000", "NONE"; kwargs...,
            ephemerides_files = [paths.spk, paths.earth_bpc],
        )
    end
end


function test_earth_axes_match_spice()
    paths = earth_ephemerides_kernel_paths()
    provider = Ephemerides.EphemerisProvider([paths.spk, paths.earth_bpc])
    H = HighFidelityEphemerisModel

    for frame in ("ITRF93", "IAU_EARTH")
        frames = H.build_ephemerides_frame_system(provider, frame)
        for date in ("2000-01-01T12:00:00", "2015-07-01T00:00:00", "2024-02-29T18:30:00", "2026-01-05T00:00:00")
            et = str2et(date)
            R = HighFidelityEphemerisModel.FrameTransformations.rotation6(frames, :ICRF, Symbol(frame), et)
            R_spice = pxform("J2000", frame, et)
            # ~1e-12 rad roundoff (large prime-meridian angle / Chebyshev evaluation)
            @test maximum(abs.(Matrix(R[1]) - R_spice)) < 1e-11
            # rotation rate (order-2 frame system) vs SPICE sxform
            S_spice = sxform("J2000", frame, et)
            @test maximum(abs.(Matrix(R[2]) - S_spice[4:6, 1:3])) < 1e-15
        end
    end

    # ECLIPJ2000 matches SPICE (no ICRF -> EME2000 frame bias, as in SPICE)
    frames = H.build_ephemerides_frame_system(provider, "ITRF93")
    et = str2et("2020-01-01T00:00:00")
    R = HighFidelityEphemerisModel.FrameTransformations.rotation3(frames, :ICRF, :ECLIPJ2000, et)
    @test maximum(abs.(Matrix(R[1]) - pxform("J2000", "ECLIPJ2000", et))) < 1e-15
end


function test_earth_axes_errors()
    paths = earth_ephemerides_kernel_paths()
    H = HighFidelityEphemerisModel
    provider_no_earth = Ephemerides.EphemerisProvider([paths.spk, paths.moon_bpc])
    # ITRF93 needs a binary Earth PCK
    err = try
        H.build_ephemerides_frame_system(provider_no_earth, "ITRF93")
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("earth_latest_high_prec.bpc", sprint(showerror, err))
    # IAU_EARTH needs no kernel
    @test H.build_ephemerides_frame_system(provider_no_earth, "IAU_EARTH") isa HighFidelityEphemerisModel.FrameTransformations.FrameSystem
    # unsupported body-fixed frames give an informative error
    err = try
        H.build_ephemerides_frame_system(provider_no_earth, "IAU_MARS")
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("ITRF93", msg) && occursin("IAU_EARTH", msg) && occursin("MOON_PA", msg)
end


function test_earth_eom_ephemerides_matches_spice()
    et0 = str2et("2026-01-05T00:00:00")
    r0 = (6378.137 + 400.0) / 6378.0
    v0 = sqrt(1 / r0)
    inc = deg2rad(51.6)
    x0 = [r0, 0.0, 0.0, 0.0, v0 * cos(inc), v0 * sin(inc)]

    for frame in ("ITRF93", "IAU_EARTH")
        params_spice = _earth_parameters(:spice, frame; et0 = et0)
        params_ephem = _earth_parameters(:ephemerides, frame; et0 = et0)

        # EOM evaluations (drag + SH + third body + SRP) agree with the SPICE backend
        for (t, x) in ((0.0, x0), (0.37, x0 .* [1.0, 1.0, 1.0, 0.99, 1.0, 1.01]),
                       (1.9, [0.3, -1.0, 0.25, 0.8, 0.2, 0.5]))
            dx_spice = HighFidelityEphemerisModel.eom_NbodySH(x, params_spice, t)
            dx_ephem = HighFidelityEphemerisModel.eom_NbodySH(x, params_ephem, t)
            @test norm(dx_ephem - dx_spice) < 1e-12 * norm(dx_spice)

            # drag-only part: isolate it by differencing with a drag-free model
            p_nodrag = _earth_parameters(:ephemerides, frame; et0 = et0, include_drag = false)
            a_drag = dx_ephem[4:6] - HighFidelityEphemerisModel.eom_NbodySH(x, p_nodrag, t)[4:6]
            @test norm(a_drag) > 0
        end

        # short propagation
        tspan = (0.0, 3 * 3600 / params_spice.TU)
        sol_spice = solve(ODEProblem(HighFidelityEphemerisModel.eom_NbodySH!, x0, tspan, params_spice),
            Vern8(), reltol = 1e-12, abstol = 1e-12)
        sol_ephem = solve(ODEProblem(HighFidelityEphemerisModel.eom_NbodySH!, x0, tspan, params_ephem),
            Vern8(), reltol = 1e-12, abstol = 1e-12)
        @test sol_ephem.retcode == SciMLBase.ReturnCode.Success
        # position difference below 1 mm
        @test norm(sol_ephem.u[end][1:3] - sol_spice.u[end][1:3]) * params_spice.DU < 1e-6

        # Jacobian through the Ephemerides backend (ForwardDiff) vs SPICE backend
        J_spice = HighFidelityEphemerisModel.eom_jacobian_fd(
            HighFidelityEphemerisModel.eom_NbodySH, x0, 0.0, params_spice, 0.0)
        J_ephem = HighFidelityEphemerisModel.eom_jacobian_fd(
            HighFidelityEphemerisModel.eom_NbodySH, x0, 0.0, params_ephem, 0.0)
        @test maximum(abs.(J_ephem - J_spice)) < 1e-10
    end
end


function test_earth_ephemerides_sun_position()
    # SPICE-free Sun for the density models, taken from the Ephemerides frame system
    et0 = str2et("2026-01-05T00:00:00")
    params = _earth_parameters(:ephemerides, "ITRF93"; et0 = et0)
    sun = HighFidelityEphemerisModel.ephemerides_sun_position_function(
        params.ephemerides_frame_system, "ITRF93")
    for et in et0 .+ (0.0, 3600.0, 86400.0)
        r_spice, _ = spkpos("SUN", et, "ITRF93", "NONE", "EARTH")
        @test norm(sun(et) - r_spice) < 1e-6 * norm(r_spice)
    end

    f_density = HighFidelityEphemerisModel.harris_priester_f_density(sun_position = sun)
    params_sun = _earth_parameters(:ephemerides, "ITRF93"; et0 = et0, f_density = f_density)
    r0 = (6378.137 + 400.0) / 6378.0
    x0 = [r0, 0.0, 0.0, 0.0, sqrt(1 / r0), 0.0]
    dx = HighFidelityEphemerisModel.eom_NbodySH(x0, params_sun, 0.0)
    dx_analytic = HighFidelityEphemerisModel.eom_NbodySH(x0, params, 0.0)
    @test all(isfinite, dx)
    # analytic vs DE440 Sun: the densities (hence drag) differ only slightly
    p_nodrag = _earth_parameters(:ephemerides, "ITRF93"; et0 = et0, include_drag = false)
    dx0 = HighFidelityEphemerisModel.eom_NbodySH(x0, p_nodrag, 0.0)
    @test norm((dx - dx0) - (dx_analytic - dx0)) < 2e-3 * norm(dx - dx0)
end


test_earth_axes_match_spice()
test_earth_axes_errors()
test_earth_eom_ephemerides_matches_spice()
test_earth_ephemerides_sun_position()
