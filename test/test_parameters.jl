"""
Test parameter types (concrete field types, backward-compatible API), allocation-free
EOM evaluation, and Ephemerides.jl thread-cache padding for prebuilt frame systems.
"""

using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

if !@isdefined(furnsh_kernels)
    include(joinpath(@__DIR__, "utils.jl"))
end

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
    furnsh_kernels()
end


function _parameters_test_kernel(kind::String, name::String)
    if haskey(ENV, "SPICE")
        return joinpath(ENV["SPICE"], kind, name)
    else
        return joinpath(@__DIR__, "../spice/test", name)
    end
end

_parameters_test_density(et, r_km) = 1e-12 * exp(-(norm(r_km) - 1737.4) / 50.0)


function _parameters_test_all(; drag::Bool = false, srp::Bool = true, kwargs...)
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    et0 = str2et("2026-01-05T00:00:00")
    DU = 3000        # Int on purpose: constructors must accept any Real
    kw = (; include_srp = srp, frame_PCPF = "MOON_PA",
        include_drag = drag, f_density = drag ? _parameters_test_density : nothing, kwargs...)
    ps = HighFidelityEphemerisModel.SpiceParameters(et0, DU, GMs, naif_ids, "J2000", "NONE"; kw...)
    pi = HighFidelityEphemerisModel.InterpParameters(et0, DU, GMs, naif_ids, "J2000", "NONE";
        interpolate_ephem_span = [et0, et0 + 3 * 86400.0], kw...)
    pe = HighFidelityEphemerisModel.EphemeridesParameters(et0, DU, GMs, naif_ids, "J2000", "NONE";
        ephemerides_files = [_parameters_test_kernel("spk", "de440.bsp"),
                             _parameters_test_kernel("pck", "moon_pa_de440_200625.bpc")], kw...)
    return ps, pi, pe
end


function _ephemerides_cache_lengths(provider)
    lens = Int[]
    for daf in provider.files
        for fname in fieldnames(typeof(daf.seglist))
            segs = getfield(daf.seglist, fname)
            segs isa AbstractVector || continue
            for seg in segs
                hasfield(typeof(seg), :cache) || continue
                caches = getfield(seg, :cache)
                caches isa AbstractVector && !isempty(caches) && push!(lens, length(caches))
            end
        end
    end
    return lens
end


function test_parameter_field_types_and_api()
    for drag in (false, true)
        for p in _parameters_test_all(; drag = drag)
            T = typeof(p)
            for name in fieldnames(T)
                name == :u && continue      # user-owned flexible vector, not read by the EOMs
                @test isconcretetype(fieldtype(T, name)) || fieldtype(T, name) <: Union{Nothing,String,HighFidelityEphemerisModel.InterpolatedTransformation}
            end
            @test p.DU === 3000.0
            @test p.TU isa Float64 && p.VU isa Float64

            # field names and virtual properties are unchanged
            for name in (:et0, :DU, :TU, :VU, :GMs, :mus, :naif_ids, :naif_frame, :abcorr,
                    :spherical_harmonics_data, :frame_PCPF, :factorial_alias, :include_srp,
                    :k_srp_cannonball, :idx_sun, :include_drag, :k_drag, :omega_atm, :f_density, :u,
                    :interpolated_ephems, :interpolated_transformation,
                    :ephemerides_provider, :ephemerides_frame_system)
                @test name in propertynames(p)
            end

            # mutation keeps working (with conversion)
            q = copy(p)
            @test typeof(q) == typeof(p)
            q.DU = 2
            @test q.DU === 2.0
            q.include_srp = !p.include_srp
            @test q.include_srp != p.include_srp
            q.k_drag = 1
            @test q.k_drag === 1.0
            q.u = [1, 2, 3]
            @test q.u == [1, 2, 3]
            q.f_density = p.f_density
            # changing the type of a type-parameter field is rejected with a clear error
            @test_throws ArgumentError (q.f_density = (et, r) -> 0.0)
            @test p.DU === 3000.0
        end
    end
end


function _eom_allocations(f!, dx, x, p, t)
    f!(dx, x, p, t)
    return @allocations f!(dx, x, p, t)
end


function test_eom_type_stability_and_allocations()
    x = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]
    dx = zeros(6)
    t = 0.01
    for drag in (false, true)
        ps, pi, pe = _parameters_test_all(; drag = drag)
        for p in (ps, pi, pe)
            @inferred HighFidelityEphemerisModel.eom_Nbody(x, p, t)
            @inferred HighFidelityEphemerisModel.eom_NbodySH(x, p, t)
            @inferred HighFidelityEphemerisModel.eom_Nbody(ForwardDiff.Dual.(x, 1.0), p, t)
        end
        # Interp/Ephemerides in-place EOMs do not allocate; with drag only the
        # `r_km::Vector` handed to the user's `f_density` callback may allocate
        nmax_alloc = drag ? 2 : 0
        @test _eom_allocations(HighFidelityEphemerisModel.eom_Nbody!, dx, x, pi, t) <= nmax_alloc
        @test _eom_allocations(HighFidelityEphemerisModel.eom_Nbody!, dx, x, pe, t) <= nmax_alloc
    end
end


function test_ephemerides_thread_caches_frame_system_only()
    files = [_parameters_test_kernel("spk", "de440.bsp"), _parameters_test_kernel("pck", "moon_pa_de440_200625.bpc")]
    provider = HighFidelityEphemerisModel.Ephemerides.EphemerisProvider(files)
    @test all(==(Threads.nthreads()), _ephemerides_cache_lengths(provider))   # unpadded provider

    frames = HighFidelityEphemerisModel.build_ephemerides_frame_system(provider, "MOON_PA")
    @test only(HighFidelityEphemerisModel.find_ephemerides_providers(frames)) === provider

    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    et0 = str2et("2026-01-05T00:00:00")
    params = HighFidelityEphemerisModel.EphemeridesParameters(
        et0, 3000.0, GMs, naif_ids, "J2000", "NONE";
        include_srp = true, frame_PCPF = "MOON_PA", ephemerides_frame_system = frames)
    @test isnothing(params.ephemerides_provider)
    @test all(>=(Threads.maxthreadid()), _ephemerides_cache_lengths(provider))

    # explicit slot count (covers configurations where maxthreadid() == nthreads())
    nslots = Threads.maxthreadid() + 2
    HighFidelityEphemerisModel.ensure_ephemerides_thread_caches!(frames; nslots = nslots)
    @test all(>=(nslots), _ephemerides_cache_lengths(provider))

    # threaded evaluation (thread IDs > nthreads() with an interactive pool) matches serial
    xs = [[1.0, 0.1 * i, 0.3, 0.5, 1.0, 0.0] for i in 1:64]
    serial = [HighFidelityEphemerisModel.eom_Nbody(x, params, 0.01) for x in xs]
    threaded = Vector{Vector{Float64}}(undef, length(xs))
    Threads.@threads for i in eachindex(xs)
        dx = zeros(6)
        HighFidelityEphemerisModel.eom_Nbody!(dx, xs[i], params, 0.01)
        threaded[i] = dx
    end
    @test threaded == serial

    prob = ODEProblem(HighFidelityEphemerisModel.eom_Nbody!, xs[1], (0.0, 0.05), params)
    eprob = EnsembleProblem(prob; prob_func = (prob, i, repeat) -> remake(prob, u0 = xs[i]))
    sols = solve(eprob, Vern7(), EnsembleThreads(); trajectories = 8, reltol = 1e-12, abstol = 1e-12)
    for i in 1:8
        sol = solve(remake(prob, u0 = xs[i]), Vern7(); reltol = 1e-12, abstol = 1e-12)
        @test sols[i].u[end] == sol.u[end]
    end

    # a frame system without any reachable provider warns
    @test_logs (:warn, r"No Ephemerides.EphemerisProvider is reachable") HighFidelityEphemerisModel.EphemeridesParameters(
        et0, 3000.0, GMs[1:1], naif_ids[1:1], "J2000", "NONE";
        ephemerides_frame_system = HighFidelityEphemerisModel.FrameTransformations.FrameSystem{2,Float64}())
end


function _axes_allocations(name)
    HighFidelityEphemerisModel.ephemerides_axes_symbol(name)
    return @allocations HighFidelityEphemerisModel.ephemerides_axes_symbol(name)
end


function test_ephemerides_axes_mapping()
    H = HighFidelityEphemerisModel
    for (name, sym) in (("J2000", :ICRF), ("j2000", :ICRF), ("ICRF", :ICRF), ("eme2000", :EME2000),
                        ("MOON_PA", :MOON_PA), ("moon_pa", :MOON_PA), ("ITRF93", :ITRF93), ("itrf93", :ITRF93),
                        ("IAU_EARTH", :IAU_EARTH), ("iau_earth", :IAU_EARTH), ("EclipJ2000", :ECLIPJ2000),
                        ("MyFrame", :MyFrame))
        @test H.ephemerides_axes_symbol(name) === sym
        @test H._ephemerides_axes(name) === sym     # EOMs use the same mapping
        @test _axes_allocations(name) == 0
    end
end


function test_ephemerides_lowercase_earth_frames()
    H = HighFidelityEphemerisModel
    naif_ids = ["399", "301", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    et0 = str2et("2026-01-05T00:00:00")
    files = [_parameters_test_kernel("spk", "de440.bsp"), _parameters_test_kernel("pck", "earth_latest_high_prec.bpc")]
    x = [1.05, 0.0, 0.01, 0.0, 1.0, 0.0]
    for frame in ("ITRF93", "IAU_EARTH")
        kw = (; include_srp = true, include_drag = true, f_density = _parameters_test_density)
        ps = H.SpiceParameters(et0, 6378.0, GMs, naif_ids, "J2000", "NONE"; frame_PCPF = frame, kw...)
        pe = H.EphemeridesParameters(et0, 6378.0, GMs, naif_ids, "j2000", "NONE";
            frame_PCPF = lowercase(frame), ephemerides_files = files, kw...)
        @test H.eom_Nbody(x, pe, 0.01) ≈ H.eom_Nbody(x, ps, 0.01) rtol = 1e-12
    end
end


function test_spherical_harmonics_field_fast_path()
    H = HighFidelityEphemerisModel
    filepath = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    x = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]
    for p in _parameters_test_all(; filepath_spherical_harmonics = filepath, nmax = 12)
        @test haskey(p.spherical_harmonics_data, "FIELD")
        # same data without the normalized field -> legacy dictionary path
        q = copy(p)
        sh = copy(p.spherical_harmonics_data)
        delete!(sh, "FIELD")
        q.spherical_harmonics_data = sh
        dx_fast = H.eom_NbodySH(x, p, 0.01)
        dx_dict = H.eom_NbodySH(x, q, 0.01)
        @test dx_fast ≈ dx_dict rtol = 1e-14
        @test dx_fast != H.eom_Nbody(x, p, 0.01)       # SH actually applied
        @inferred H.eom_NbodySH(x, p, 0.01)
        @inferred H.eom_NbodySH(ForwardDiff.Dual.(x, 1.0), p, 0.01)
        J_fast = H.eom_jacobian_fd(H.eom_NbodySH, x, 0.0, p, 0.01)
        J_dict = H.eom_jacobian_fd(H.eom_NbodySH, x, 0.0, q, 0.01)
        @test J_fast ≈ J_dict rtol = 1e-12
    end
end


function _sh_test_params(nmax = 6)
    filepath = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    _, pi, _ = _parameters_test_all(; filepath_spherical_harmonics = filepath, nmax = nmax)
    return pi
end

# replace one entry of the SH dictionary on a copy of `p` (the copy shares nothing new)
function _with_sh(p, changes::Pair...)
    q = copy(p)
    sh = copy(p.spherical_harmonics_data)
    for (k, v) in changes
        sh[k] = v
    end
    q.spherical_harmonics_data = sh
    return q
end


function test_spherical_harmonics_coefficient_changes()
test_abcorr_canonical_spelling()
    H = HighFidelityEphemerisModel
    x = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]
    t = 0.01

    # replacing Cnm/Snm by other dictionaries is honoured (not silently ignored)
    p = _sh_test_params()
    zero_coefs = Dict{Tuple{Int,Int},Float64}(k => 0.0 for k in keys(p.spherical_harmonics_data["Cnm"]))
    q = _with_sh(p, "Cnm" => zero_coefs, "Snm" => zero_coefs)
    @test H.eom_NbodySH(x, q, t) ≈ H.eom_Nbody(x, p, t) rtol = 1e-15
    @test H.eom_NbodySH(x, p, t) != H.eom_Nbody(x, p, t)
    # replacing only one of the two sets also takes effect
    q = _with_sh(p, "Snm" => zero_coefs)
    Cdict = Dict(p.spherical_harmonics_data["Cnm"])          # un-normalized copy of C
    q_ref = _with_sh(p, "Cnm" => Cdict, "Snm" => zero_coefs)
    @test H.eom_NbodySH(x, q, t) ≈ H.eom_NbodySH(x, q_ref, t) rtol = 1e-13
    @test H.eom_NbodySH(x, q, t) != H.eom_NbodySH(x, p, t)

    # in-place mutation of the coefficient view writes through to the field
    p = _sh_test_params()                                    # fresh field (mutated below)
    sh = p.spherical_harmonics_data
    a_before = H.eom_NbodySH(x, p, t)
    c20 = sh["Cnm"][2, 0]
    sh["Cnm"][2, 0] = 0.0
    @test sh["Cnm"][2, 0] == 0.0
    @test sh["Cnm"][(2, 0)] == 0.0
    a_after = H.eom_NbodySH(x, p, t)
    @test a_after != a_before
    q_ref = _with_sh(p, "Cnm" => Dict(sh["Cnm"]), "Snm" => Dict(sh["Snm"]))   # dictionary path
    @test a_after ≈ H.eom_NbodySH(x, q_ref, t) rtol = 1e-12
    sh["Cnm"][(2, 0)] = c20
    @test sh["Cnm"][2, 0] ≈ c20 rtol = 1e-15
    @test H.eom_NbodySH(x, p, t) ≈ a_before rtol = 1e-14
    @test_throws KeyError (sh["Cnm"][1, 0] = 1.0)
    @test_throws ArgumentError (sh["Cnm"][2, 0] = ForwardDiff.Dual(1.0, 1.0))

    # sensitivity w.r.t. a coefficient, through Dual-valued coefficient dictionaries
    p = _sh_test_params()
    Cbase = Dict(p.spherical_harmonics_data["Cnm"])
    Sbase = Dict(p.spherical_harmonics_data["Snm"])
    function a_of_c20(c)
        C = Dict{Tuple{Int,Int},typeof(c)}(k => (k == (2, 0) ? c : v) for (k, v) in Cbase)
        S = Dict{Tuple{Int,Int},typeof(c)}(k => v for (k, v) in Sbase)
        # the state carries the same Dual type as the coefficients (type-stable EOM)
        return H.eom_NbodySH(x .+ zero(c), _with_sh(p, "Cnm" => C, "Snm" => S), t)
    end
    c20 = Cbase[(2, 0)]
    @test a_of_c20(c20) ≈ H.eom_NbodySH(x, p, t) rtol = 1e-12
    d_ad = ForwardDiff.derivative(a_of_c20, c20)
    h = 1e-3 * abs(c20)
    d_fd = (a_of_c20(c20 + h) - a_of_c20(c20 - h)) / (2h)
    @test maximum(abs, d_ad[4:6]) > 0
    @test d_ad ≈ d_fd rtol = 1e-6
    @test H.spherical_harmonics_accel_PCPF([1800.0, 100.0, 300.0],
        Dict{Tuple{Int,Int},ForwardDiff.Dual{Nothing,Float64,1}}(k => ForwardDiff.Dual(v, k == (2, 0)) for (k, v) in Cbase),
        Sbase, p.spherical_harmonics_data["GM"], p.spherical_harmonics_data["REFERENCE RADIUS"], 6) isa
        Vector{<:ForwardDiff.Dual}

    # sensitivity w.r.t. the field's GM (Dual GM goes through the generic path)
    GM0 = p.spherical_harmonics_data["GM"]
    a_of_GM(gm) = H.eom_NbodySH(x .+ zero(gm), _with_sh(p, "GM" => gm), t)
    @test a_of_GM(GM0) == H.eom_NbodySH(x, p, t)
    d_ad = ForwardDiff.derivative(a_of_GM, GM0)
    h = 1e-4 * GM0
    d_fd = (a_of_GM(GM0 + h) - a_of_GM(GM0 - h)) / (2h)
    @test maximum(abs, d_ad[4:6]) > 0
    @test d_ad ≈ d_fd rtol = 1e-7
    # a Dual GM with a Float64 state gives a clear error instead of a MethodError
    @test_throws ArgumentError H.eom_NbodySH(x, _with_sh(p, "GM" => ForwardDiff.Dual(GM0, 1.0)), t)
end


# SPICE ignores case and ASCII blanks in `abcorr` (and nothing else), so every backend
# stores one canonical spelling that SPICE treats identically.
function test_abcorr_canonical_spelling()
    naif_ids = ["399", "301"]
    GMs = [1.0, 1.0e-6]
    params = HighFidelityEphemerisModel.SpiceParameters(0.0, 1.0, GMs, naif_ids, "J2000", " lt + s ")
    @test params.abcorr == "LT+S"
    params.abcorr = "cn+s "
    @test params.abcorr == "CN+S"
    params.abcorr = SubString("xlt + s", 1)
    @test params.abcorr == "XLT+S"

    canonical = HighFidelityEphemerisModel._canonical_abcorr
    # tabs, other whitespace and non-ASCII letters are kept, so SPICE still rejects them
    @test canonical("lt\t+s") == "LT\t+S"
    @test canonical("LT\u00a0+S") == "LT\u00a0+S"
    @test canonical("LT+ſ") == "LT+ſ"
    for abcorr in ("NONE", "LT", "LT+S", "CN", "CN+S", "XLT", "XLT+S", "XCN", "XCN+S")
        for spelling in (abcorr, lowercase(abcorr), " " * join(collect(lowercase(abcorr)), " ") * " ")
            @test canonical(spelling) == abcorr
            # SPICE gives identical results for the given and the canonical spelling
            @test spkpos("301", 1e8, "J2000", spelling, "399")[1] ==
                  spkpos("301", 1e8, "J2000", canonical(spelling), "399")[1]
        end
    end
    for abcorr in ("LT\t+S", "LT\u00a0+S", "LT+ſ", "NONE\t")
        @test_throws SPICE.SpiceError spkpos("301", 1e8, "J2000", abcorr, "399")
        @test_throws SPICE.SpiceError spkpos("301", 1e8, "J2000", canonical(abcorr), "399")
    end

    is_none = HighFidelityEphemerisModel._is_none_abcorr
    for spelling in ("NONE", "none", " n o N e ", SubString("xnone", 2))
        @test is_none(spelling)
        @test (@allocations is_none(spelling)) == 0
    end
    for spelling in ("", " ", "NON", "NONEE", "NONE\t", "NONE+S", "LT", "ＮＯＮＥ")
        @test !is_none(spelling)
    end
end


test_parameter_field_types_and_api()
test_eom_type_stability_and_allocations()
test_ephemerides_thread_caches_frame_system_only()
test_ephemerides_axes_mapping()
test_spherical_harmonics_field_fast_path()
test_ephemerides_lowercase_earth_frames()
test_spherical_harmonics_coefficient_changes()
test_abcorr_canonical_spelling()
