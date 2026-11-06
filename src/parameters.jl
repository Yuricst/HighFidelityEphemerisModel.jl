"""Parameters for HFEM dynamics backends."""


abstract type HighFidelityEphemerisModelParameters end
const AbstractHFEMParameters = HighFidelityEphemerisModelParameters


# Field-type notes (type stability of the dynamics):
#
# - `DU`, `TU`, `VU` are stored as `Float64`; constructors accept any `Real` and
#   assigning e.g. an `Int` later still works (it is converted).
# - Fields whose concrete type depends on the user's inputs are type parameters:
#   `spherical_harmonics_data` (`SH`), `factorial_alias` (`FA`), `f_density`
#   (`FD`) and, for `EphemeridesParameters`, `ephemerides_backend` (`EB`). This
#   lets the equations of motion compile to type-stable code. These fields can
#   still be reassigned, but only to a value of the same type; to switch e.g.
#   from no density model to a density callback, construct new parameters.
# - `u` is a user-owned control/parameter vector that is not read by any of the
#   package's equations of motion. It is deliberately kept as a flexible
#   `Vector` (any element type, e.g. ForwardDiff duals), so user code that reads
#   it in a hot loop should pass `params.u` through a function barrier.

"""
    SpiceParameters{SH,FA,FD}

Parameters for the N-body (+ spherical harmonics, SRP, drag) dynamics in which body
positions and frame rotations are evaluated with SPICE (`spkpos`, `pxform`) at every
EOM call. Requires the corresponding kernels to be furnished.

# Construction

    SpiceParameters(et0, DU, GMs, naif_ids, naif_frame = "J2000", abcorr = "NONE"; kwargs...)

`et0` is the reference epoch (TDB seconds past J2000), `DU` the canonical length unit
in km (any `Real`; stored as `Float64`), `GMs` the gravitational parameters (km³/s²) of
the bodies `naif_ids`, the first of which is the central body. Common keyword arguments:
`filepath_spherical_harmonics`, `nmax`, `frame_PCPF`, `include_srp`, `srp_Cr`, `srp_Am`,
`srp_P0`, `include_drag`, `drag_Cd`, `drag_Am`, `f_density`, `omega_atm`, `nu`,
`use_canonical_scales`.

Fields can be read and assigned as usual; fields whose type is a type parameter
(`spherical_harmonics_data`, `factorial_alias`, `f_density`) only accept values of
the same type (construct new parameters to change e.g. the density model type).
`copy(params)` gives an independently mutable copy that shares the heavy data.
"""
mutable struct SpiceParameters{SH,FA,FD} <: HighFidelityEphemerisModelParameters
    et0::Float64
    DU::Float64
    TU::Float64
    VU::Float64

    GMs::Vector{Float64}
    mus::Vector{Float64}
    naif_ids::Vector{String}
    naif_frame::String
    abcorr::String
    spherical_harmonics_data::SH
    frame_PCPF::Union{Nothing,String}
    factorial_alias::FA

    include_srp::Bool
    k_srp_cannonball::Float64
    idx_sun::Int

    include_drag::Bool
    k_drag::Float64
    omega_atm::Vector{Float64}
    f_density::FD

    u::Vector
end


"""
    InterpParameters{SH,FA,FD}

Parameters for the N-body (+ spherical harmonics, SRP, drag) dynamics in which body
positions and the inertial-to-body-fixed rotation are spline-interpolated from SPICE
samples taken at construction (`interpolate_ephem_span`, `interpolation_time_step`).

# Construction

    InterpParameters(et0, DU, GMs, naif_ids, naif_frame = "J2000", abcorr = "NONE"; kwargs...)

`et0` is the reference epoch (TDB seconds past J2000), `DU` the canonical length unit
in km (any `Real`; stored as `Float64`), `GMs` the gravitational parameters (km³/s²) of
the bodies `naif_ids`, the first of which is the central body. Common keyword arguments:
`filepath_spherical_harmonics`, `nmax`, `frame_PCPF`, `include_srp`, `srp_Cr`, `srp_Am`,
`srp_P0`, `include_drag`, `drag_Cd`, `drag_Am`, `f_density`, `omega_atm`, `nu`,
`use_canonical_scales`.
Backend-specific keyword arguments: `interpolate_ephem_span::Vector{Float64}` (required), `interpolation_time_step`.

Fields can be read and assigned as usual; fields whose type is a type parameter
(`spherical_harmonics_data`, `factorial_alias`, `f_density`) only accept values of
the same type (construct new parameters to change e.g. the density model type).
`copy(params)` gives an independently mutable copy that shares the heavy data.
"""
mutable struct InterpParameters{SH,FA,FD} <: HighFidelityEphemerisModelParameters
    et0::Float64
    DU::Float64
    TU::Float64
    VU::Float64

    GMs::Vector{Float64}
    mus::Vector{Float64}
    naif_ids::Vector{String}
    naif_frame::String
    abcorr::String
    interpolated_ephems::Vector{InterpolatedEphemeris}
    spherical_harmonics_data::SH
    frame_PCPF::Union{Nothing,String}
    factorial_alias::FA
    interpolated_transformation::Union{Nothing,InterpolatedTransformation}

    include_srp::Bool
    k_srp_cannonball::Float64
    idx_sun::Int

    include_drag::Bool
    k_drag::Float64
    omega_atm::Vector{Float64}
    f_density::FD

    u::Vector
end


"""
    EphemeridesParameters{SH,FA,EB,FD}

Parameters for the N-body (+ spherical harmonics, SRP, drag) dynamics in which body
positions and frame rotations are evaluated with Julia-native Ephemerides.jl /
FrameTransformations.jl (thread-safe, ForwardDiff-compatible in time).

# Construction

    EphemeridesParameters(et0, DU, GMs, naif_ids, naif_frame = "J2000", abcorr = "NONE"; kwargs...)

`et0` is the reference epoch (TDB seconds past J2000), `DU` the canonical length unit
in km (any `Real`; stored as `Float64`), `GMs` the gravitational parameters (km³/s²) of
the bodies `naif_ids`, the first of which is the central body. Common keyword arguments:
`filepath_spherical_harmonics`, `nmax`, `frame_PCPF`, `include_srp`, `srp_Cr`, `srp_Am`,
`srp_P0`, `include_drag`, `drag_Cd`, `drag_Am`, `f_density`, `omega_atm`, `nu`,
`use_canonical_scales`.
Backend-specific keyword arguments: one of `ephemerides_files`, `ephemerides_provider`, or a prebuilt `ephemerides_frame_system` (the per-thread caches of every Ephemerides.jl provider reachable from it are padded to `Threads.maxthreadid()`).

`abcorr` must be `"NONE"` (case and blanks are ignored): the Ephemerides.jl /
FrameTransformations.jl backend only returns geometric states, so any aberration
correction throws an `ArgumentError`, both at construction and on assignment to
`params.abcorr`. Use `SpiceParameters` for aberration-corrected dynamics.

Fields can be read and assigned as usual; fields whose type is a type parameter
(`spherical_harmonics_data`, `factorial_alias`, `f_density`, `ephemerides_backend`) only accept values of
the same type (construct new parameters to change e.g. the density model type).
`copy(params)` gives an independently mutable copy that shares the heavy data.
"""
mutable struct EphemeridesParameters{SH,FA,EB,FD} <: HighFidelityEphemerisModelParameters
    et0::Float64
    DU::Float64
    TU::Float64
    VU::Float64

    GMs::Vector{Float64}
    mus::Vector{Float64}
    naif_ids::Vector{String}
    naif_frame::String
    abcorr::String
    spherical_harmonics_data::SH
    frame_PCPF::Union{Nothing,String}
    factorial_alias::FA
    ephemerides_backend::EB

    include_srp::Bool
    k_srp_cannonball::Float64
    idx_sun::Int

    include_drag::Bool
    k_drag::Float64
    omega_atm::Vector{Float64}
    f_density::FD

    u::Vector
end


# Fields whose type is a struct type parameter (see notes above).
const _HFEM_PARAMETRIC_FIELDS = (:spherical_harmonics_data, :factorial_alias, :f_density, :ephemerides_backend)


function Base.setproperty!(params::HighFidelityEphemerisModelParameters, name::Symbol, value)
    T = fieldtype(typeof(params), name)
    if name in _HFEM_PARAMETRIC_FIELDS && !(value isa T)
        throw(ArgumentError(
            "cannot assign a value of type $(typeof(value)) to `$name`: these parameters were " *
            "constructed with `$name::$T`, and this field's type is fixed so that the equations " *
            "of motion are type-stable. Construct new parameters with the desired `$name` instead."
        ))
    end
    if name === :abcorr
        value = convert(String, value)
        # validate the value as given, so the error shows what the user wrote
        params isa EphemeridesParameters && _validate_ephemerides_abcorr(value)
        value = _canonical_abcorr(value)
    end
    return setfield!(params, name, convert(T, value))
end


"""
    copy(params)

Copy of HFEM parameters that is safe to mutate independently of the original.

Scalars (e.g. `k_srp_cannonball`, `k_drag`) and the small mutable vectors
(`GMs`, `mus`, `naif_ids`, `omega_atm`, `u`) are independent on the copy.
Heavy or read-only data (ephemerides provider/frame system, interpolated
ephemerides/transformation, spherical harmonics coefficients, `f_density`) are
shared with the original and must not be mutated in place.

Intended for per-trajectory parameter changes in an `EnsembleProblem`
`prob_func`, together with `safetycopy = false` to avoid deep-copying the
ephemeris data for every trajectory.
"""
function Base.copy(params::HighFidelityEphemerisModelParameters)
    T = typeof(params)
    return T(ntuple(fieldcount(T)) do i
        v = getfield(params, i)
        fieldname(T, i) in (:GMs, :mus, :naif_ids, :omega_atm, :u) ? copy(v) : v
    end...)
end


function Base.show(io::IO, params::HighFidelityEphemerisModelParameters)
    println(io, string(nameof(typeof(params))))
    @printf(io, "    et0              : %s (et = %1.8f)\n", et2utc(params.et0, "ISOC", 3), params.et0)
    @printf(io, "    DU               : %1.8f\n", params.DU)
    @printf(io, "    TU               : %1.8f\n", params.TU)
    @printf(io, "    VU               : %1.8f\n", params.VU)
    @printf(io, "    include_srp      : %s\n", params.include_srp)
    if params.include_srp
        @printf(io, "    k_srp_cannonball : %1.8f\n", params.k_srp_cannonball)
        @printf(io, "    idx_sun          : %d\n", params.idx_sun)
    end
    @printf(io, "    include_drag     : %s\n", params.include_drag)
    if params.include_drag
        @printf(io, "    k_drag           : %1.8f\n", params.k_drag)
        @printf(io, "    omega_atm        : [%1.8f, %1.8f, %1.8f]\n", params.omega_atm...)
    end
end


function Base.getproperty(params::Union{SpiceParameters,InterpParameters}, name::Symbol)
    if name == :ephemerides_provider || name == :ephemerides_frame_system
        return nothing
    elseif params isa SpiceParameters && (name == :interpolated_ephems || name == :interpolated_transformation)
        return nothing
    else
        return getfield(params, name)
    end
end


function Base.getproperty(params::EphemeridesParameters, name::Symbol)
    if name == :ephemerides_provider
        return getfield(getfield(params, :ephemerides_backend), :provider)
    elseif name == :ephemerides_frame_system
        return getfield(getfield(params, :ephemerides_backend), :frame_system)
    elseif name == :interpolated_ephems || name == :interpolated_transformation
        return nothing
    else
        return getfield(params, name)
    end
end


function Base.propertynames(params::SpiceParameters; private::Bool = false)
    return (fieldnames(typeof(params))..., :interpolated_ephems, :interpolated_transformation,
        :ephemerides_provider, :ephemerides_frame_system)
end


function Base.propertynames(params::InterpParameters; private::Bool = false)
    return (fieldnames(typeof(params))..., :ephemerides_provider, :ephemerides_frame_system)
end


function Base.propertynames(params::EphemeridesParameters; private::Bool = false)
    return (fieldnames(typeof(params))..., :ephemerides_provider, :ephemerides_frame_system,
        :interpolated_ephems, :interpolated_transformation)
end


function _hfem_common_fields(
    et0::Float64,
    DU::Real,
    GMs::Vector{Float64},
    naif_ids::Vector{String},
    naif_frame::String,
    abcorr::String;
    filepath_spherical_harmonics::Union{Nothing,String} = nothing,
    nmax::Int = 4,
    frame_PCPF::Union{Nothing,String} = nothing,
    include_srp::Bool = false,
    srp_Cr::Float64 = 1.15,
    srp_Am::Float64 = 0.002,
    srp_P0::Float64 = 4.56e-6,
    include_drag::Bool = false,
    drag_Cd::Float64 = 2.2,
    drag_Am::Float64 = 0.01,
    f_density::Union{Nothing,Function} = nothing,
    omega_atm::Vector{Float64} = [0.0, 0.0, 7.2921159e-5],
    nu::Int = 4,
    use_canonical_scales::Bool = true,
)
    if use_canonical_scales
        DU = Float64(DU)
        VU = sqrt(GMs[1] / DU)
        TU = DU / VU
        mus = GMs / GMs[1]
    else
        DU, TU, VU = 1.0, 1.0, 1.0
        mus = GMs
    end

    factorial_alias = nmax <= 9 ? factorial : factorial_safe
    if !isnothing(filepath_spherical_harmonics)
        spherical_harmonics_data = load_spherical_harmonics(
            filepath_spherical_harmonics, nmax, true, factorial_alias
        )
    else
        spherical_harmonics_data = nothing
    end

    k_srp_cannonball = get_srp_cannonball_coefficient(DU, TU, srp_Cr, srp_Am, srp_P0)
    if "10" in naif_ids
        idx_sun = findfirst(x -> x == "10", naif_ids)
    elseif include_srp
        error("NAIF ID \"10\" (Sun) must be provided in `naif_ids` when SRP is included")
    else
        k_srp_cannonball = 0.0
        idx_sun = 0
    end

    if include_drag
        isnothing(f_density) && error("`f_density` must be provided when drag is included")
        isnothing(frame_PCPF) && error("`frame_PCPF` must be provided when drag is included")
        k_drag = get_drag_coefficient(DU, TU, VU, drag_Cd, drag_Am)
    else
        k_drag = 0.0
    end

    abcorr = _canonical_abcorr(abcorr)

    return (; et0, DU, TU, VU, GMs, mus, naif_ids, naif_frame, abcorr,
        spherical_harmonics_data, frame_PCPF, factorial_alias,
        include_srp, k_srp_cannonball, idx_sun,
        include_drag, k_drag, omega_atm, f_density, u = zeros(nu))
end


function _interpolate_ephemerides(naif_ids, naif_frame, abcorr, center_id, TU, span, dt, frame_PCPF)
    isnothing(span) && error("InterpParameters require `interpolate_ephem_span`.")

    # N intervals of at most `dt` -> N + 1 nodes, so the node spacing never exceeds `dt`
    N_interp = Int(ceil((span[2] - span[1]) / dt)) + 1
    ets_interp = range(span[1], span[2], N_interp)

    interpolated_ephems = InterpolatedEphemeris[]
    for ID in naif_ids[2:end]
        rvs_interp = hcat([spkezr(ID, et, naif_frame, abcorr, center_id)[1] for et in ets_interp]...)
        push!(interpolated_ephems, InterpolatedEphemeris(ID, ets_interp, rvs_interp, false, TU))
    end

    interpolated_transformation = isnothing(frame_PCPF) ? nothing :
        InterpolatedTransformation(ets_interp, naif_frame, frame_PCPF, false, TU)

    return interpolated_ephems, interpolated_transformation
end


# SPICE ignores case and ASCII blanks (' ') in `abcorr`, and nothing else: tabs, other
# whitespace and non-ASCII letters make SPICE reject the string. Every backend stores
# one canonical spelling (e.g. " lt + s " -> "LT+S") that SPICE treats identically, so a
# string SPICE would reject is never turned into a valid one. The canonical "NONE" also
# keeps the per-evaluation Ephemerides check on its `== "NONE"` fast path.
@inline _ascii_uppercase(c::Char) = 'a' <= c <= 'z' ? c - ('a' - 'A') : c

_canonical_abcorr(abcorr::AbstractString) = map(_ascii_uppercase, replace(abcorr, ' ' => ""))


# Non-allocating `_canonical_abcorr(abcorr) == "NONE"`.
function _is_none_abcorr(abcorr::AbstractString)
    k = 0
    for c in abcorr
        c == ' ' && continue
        k += 1
        (k <= 4 && _ascii_uppercase(c) == ('N', 'O', 'N', 'E')[k]) || return false
    end
    return k == 4
end


function _validate_ephemerides_abcorr(abcorr::AbstractString)
    _is_none_abcorr(abcorr) && return nothing

    throw(ArgumentError(
        "EphemeridesParameters only support abcorr = \"NONE\" (got \"$abcorr\") because the " *
        "Ephemerides.jl/FrameTransformations backend returns geometric states. " *
        "Use SpiceParameters or InterpParameters for aberration-corrected trajectories."
    ))
end


function _make_ephemerides_backend(ephemerides_provider, ephemerides_files, ephemerides_frame_system, frame_PCPF)
    if !isnothing(ephemerides_provider) && !isnothing(ephemerides_files)
        error("Provide either `ephemerides_provider` or `ephemerides_files`, not both.")
    end

    if isnothing(ephemerides_provider) && !isnothing(ephemerides_files)
        ephemerides_provider = Ephemerides.EphemerisProvider(ephemerides_files)
    end

    if isnothing(ephemerides_provider) && isnothing(ephemerides_frame_system)
        error("EphemeridesParameters require `ephemerides_provider`, `ephemerides_files`, or `ephemerides_frame_system`.")
    end

    ensure_ephemerides_thread_caches!(ephemerides_provider)

    if isnothing(ephemerides_frame_system)
        ephemerides_frame_system = build_ephemerides_frame_system(ephemerides_provider, frame_PCPF)
    else
        # A user-supplied frame system is queried directly by the EOMs, so the
        # provider(s) captured by its point/axes functions must be padded too,
        # whether or not `ephemerides_provider` was also given.
        providers = find_ephemerides_providers(ephemerides_frame_system)
        for provider in providers
            ensure_ephemerides_thread_caches!(provider)
        end
        if isempty(providers) && isnothing(ephemerides_provider) &&
                ephemerides_frame_system isa FrameTransformations.FrameSystem
            @warn "No Ephemerides.EphemerisProvider is reachable from `ephemerides_frame_system`, so " *
                "its per-thread segment caches cannot be padded to `Threads.maxthreadid()`. If the " *
                "frame system is backed by an Ephemerides.jl provider, also pass it as " *
                "`ephemerides_provider` to make multithreaded use (e.g. `EnsembleThreads`) safe." maxlog = 1
        end
    end

    return EphemeridesBackend(ephemerides_provider, ephemerides_frame_system)
end


function _construct_hfem_parameters(
    backend::Symbol,
    et0::Float64,
    DU::Real,
    GMs::Vector{Float64},
    naif_ids::Vector{String},
    naif_frame::String = "J2000",
    abcorr::String = "NONE";
    filepath_spherical_harmonics::Union{Nothing,String} = nothing,
    nmax::Int = 4,
    frame_PCPF::Union{Nothing,String} = nothing,
    interpolate_ephem_span::Union{Nothing,Vector{Float64}} = nothing,
    interpolation_time_step::Real = 3600.0,
    ephemerides_provider = nothing,
    ephemerides_files::Union{Nothing,String,Vector{String}} = nothing,
    ephemerides_frame_system = nothing,
    include_srp::Bool = false,
    srp_Cr::Float64 = 1.15,
    srp_Am::Float64 = 0.002,
    srp_P0::Float64 = 4.56e-6,
    include_drag::Bool = false,
    drag_Cd::Float64 = 2.2,
    drag_Am::Float64 = 0.01,
    f_density::Union{Nothing,Function} = nothing,
    omega_atm::Vector{Float64} = [0.0, 0.0, 7.2921159e-5],
    nu::Int = 4,
    use_canonical_scales::Bool = true,
)
    uses_interp = !isnothing(interpolate_ephem_span)
    uses_ephem = !isnothing(ephemerides_provider) || !isnothing(ephemerides_files) || !isnothing(ephemerides_frame_system)

    if backend == :auto
        uses_interp && uses_ephem && error("Choose either interpolation or Ephemerides backend inputs, not both.")
        backend = uses_ephem ? :ephemerides : uses_interp ? :interp : :spice
    elseif backend == :spice
        (uses_interp || uses_ephem) && error("SpiceParameters do not accept interpolation or Ephemerides backend inputs.")
    elseif backend == :interp
        uses_ephem && error("InterpParameters do not accept Ephemerides backend inputs.")
        !uses_interp && error("InterpParameters require `interpolate_ephem_span`.")
    elseif backend == :ephemerides
        uses_interp && error("EphemeridesParameters do not accept interpolation backend inputs.")
        !uses_ephem && error("EphemeridesParameters require `ephemerides_provider`, `ephemerides_files`, or `ephemerides_frame_system`.")
    else
        error("Unsupported HFEM parameter backend: $backend")
    end

    c = _hfem_common_fields(et0, DU, GMs, naif_ids, naif_frame, abcorr;
        filepath_spherical_harmonics, nmax, frame_PCPF,
        include_srp, srp_Cr, srp_Am, srp_P0,
        include_drag, drag_Cd, drag_Am, f_density, omega_atm, nu,
        use_canonical_scales)

    if backend == :spice
        return SpiceParameters{typeof(c.spherical_harmonics_data),typeof(c.factorial_alias),typeof(c.f_density)}(
            c.et0, c.DU, c.TU, c.VU,
            c.GMs, c.mus, c.naif_ids, c.naif_frame, c.abcorr,
            c.spherical_harmonics_data, c.frame_PCPF, c.factorial_alias,
            c.include_srp, c.k_srp_cannonball, c.idx_sun,
            c.include_drag, c.k_drag, c.omega_atm, c.f_density, c.u)
    elseif backend == :interp
        interpolated_ephems, interpolated_transformation = _interpolate_ephemerides(
            c.naif_ids, c.naif_frame, c.abcorr, c.naif_ids[1], c.TU,
            interpolate_ephem_span, interpolation_time_step, c.frame_PCPF)

        return InterpParameters{typeof(c.spherical_harmonics_data),typeof(c.factorial_alias),typeof(c.f_density)}(
            c.et0, c.DU, c.TU, c.VU,
            c.GMs, c.mus, c.naif_ids, c.naif_frame, c.abcorr,
            interpolated_ephems, c.spherical_harmonics_data, c.frame_PCPF,
            c.factorial_alias, interpolated_transformation,
            c.include_srp, c.k_srp_cannonball, c.idx_sun,
            c.include_drag, c.k_drag, c.omega_atm, c.f_density, c.u)
    else
        _validate_ephemerides_abcorr(abcorr)   # as given (`c.abcorr` is canonicalized)

        ephemerides_backend = _make_ephemerides_backend(
            ephemerides_provider, ephemerides_files, ephemerides_frame_system, c.frame_PCPF)

        return EphemeridesParameters{typeof(c.spherical_harmonics_data),typeof(c.factorial_alias),
                typeof(ephemerides_backend),typeof(c.f_density)}(
            c.et0, c.DU, c.TU, c.VU,
            c.GMs, c.mus, c.naif_ids, c.naif_frame, c.abcorr,
            c.spherical_harmonics_data, c.frame_PCPF, c.factorial_alias,
            ephemerides_backend,
            c.include_srp, c.k_srp_cannonball, c.idx_sun,
            c.include_drag, c.k_drag, c.omega_atm, c.f_density, c.u)
    end
end


SpiceParameters(args...; kwargs...) = _construct_hfem_parameters(:spice, args...; kwargs...)
InterpParameters(args...; kwargs...) = _construct_hfem_parameters(:interp, args...; kwargs...)
EphemeridesParameters(args...; kwargs...) = _construct_hfem_parameters(:ephemerides, args...; kwargs...)


"""
    HighFidelityEphemerisModelParameters(args...; kwargs...)

Backward-compatible constructor that returns the concrete parameter type implied
by the supplied backend inputs. With no backend inputs it returns
`SpiceParameters`; with `interpolate_ephem_span` it returns `InterpParameters`;
with Ephemerides.jl provider/files/frame-system inputs it returns
`EphemeridesParameters`.
"""
HighFidelityEphemerisModelParameters(args...; backend::Symbol = :auto, kwargs...) =
    _construct_hfem_parameters(backend, args...; kwargs...)