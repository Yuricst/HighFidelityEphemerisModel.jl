"""Parameters for HFEM dynamics backends."""


abstract type HighFidelityEphemerisModelParameters end
const AbstractHFEMParameters = HighFidelityEphemerisModelParameters


mutable struct SpiceParameters <: HighFidelityEphemerisModelParameters
    et0::Float64
    DU::Real
    TU::Real
    VU::Real

    GMs::Vector{Float64}
    mus::Vector{Float64}
    naif_ids::Vector{String}
    naif_frame::String
    abcorr::String
    spherical_harmonics_data::Union{Nothing,Dict}
    frame_PCPF::Union{Nothing,String}
    factorial_alias::Function

    include_srp::Bool
    k_srp_cannonball::Float64
    idx_sun::Int

    include_drag::Bool
    k_drag::Float64
    omega_atm::Vector{Float64}
    f_density::Union{Nothing,Function}

    u::Vector
end


mutable struct InterpParameters <: HighFidelityEphemerisModelParameters
    et0::Float64
    DU::Real
    TU::Real
    VU::Real

    GMs::Vector{Float64}
    mus::Vector{Float64}
    naif_ids::Vector{String}
    naif_frame::String
    abcorr::String
    interpolated_ephems::Vector{InterpolatedEphemeris}
    spherical_harmonics_data::Union{Nothing,Dict}
    frame_PCPF::Union{Nothing,String}
    factorial_alias::Function
    interpolated_transformation::Union{Nothing,InterpolatedTransformation}

    include_srp::Bool
    k_srp_cannonball::Float64
    idx_sun::Int

    include_drag::Bool
    k_drag::Float64
    omega_atm::Vector{Float64}
    f_density::Union{Nothing,Function}

    u::Vector
end


mutable struct EphemeridesParameters <: HighFidelityEphemerisModelParameters
    et0::Float64
    DU::Real
    TU::Real
    VU::Real

    GMs::Vector{Float64}
    mus::Vector{Float64}
    naif_ids::Vector{String}
    naif_frame::String
    abcorr::String
    spherical_harmonics_data::Union{Nothing,Dict}
    frame_PCPF::Union{Nothing,String}
    factorial_alias::Function
    ephemerides_backend

    include_srp::Bool
    k_srp_cannonball::Float64
    idx_sun::Int

    include_drag::Bool
    k_drag::Float64
    omega_atm::Vector{Float64}
    f_density::Union{Nothing,Function}

    u::Vector
end


function Base.show(io::IO, params::HighFidelityEphemerisModelParameters)
    println(io, string(typeof(params)))
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
        @error "NAIF ID \"10\" (Sun) must be provided when SRP is included"
        idx_sun = 0
    else
        k_srp_cannonball = 0.0
        idx_sun = 0
    end

    if include_drag
        if isnothing(f_density)
            @error "f_density must be provided when drag is included"
        end
        if isnothing(frame_PCPF)
            @error "frame_PCPF must be provided when drag is included"
        end
        k_drag = get_drag_coefficient(DU, TU, VU, drag_Cd, drag_Am)
    else
        k_drag = 0.0
    end

    return (; et0, DU, TU, VU, GMs, mus, naif_ids, naif_frame, abcorr,
        spherical_harmonics_data, frame_PCPF, factorial_alias,
        include_srp, k_srp_cannonball, idx_sun,
        include_drag, k_drag, omega_atm, f_density, u = zeros(nu))
end


function _interpolate_ephemerides(naif_ids, naif_frame, abcorr, center_id, TU, span, dt, frame_PCPF)
    isnothing(span) && error("InterpParameters require `interpolate_ephem_span`.")

    N_interp = Int(ceil((span[2] - span[1]) / dt))
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

    if isnothing(ephemerides_frame_system)
        ephemerides_frame_system = build_ephemerides_frame_system(ephemerides_provider, frame_PCPF)
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
        return SpiceParameters(c.et0, c.DU, c.TU, c.VU,
            c.GMs, c.mus, c.naif_ids, c.naif_frame, c.abcorr,
            c.spherical_harmonics_data, c.frame_PCPF, c.factorial_alias,
            c.include_srp, c.k_srp_cannonball, c.idx_sun,
            c.include_drag, c.k_drag, c.omega_atm, c.f_density, c.u)
    elseif backend == :interp
        interpolated_ephems, interpolated_transformation = _interpolate_ephemerides(
            c.naif_ids, c.naif_frame, c.abcorr, c.naif_ids[1], c.TU,
            interpolate_ephem_span, interpolation_time_step, c.frame_PCPF)

        return InterpParameters(c.et0, c.DU, c.TU, c.VU,
            c.GMs, c.mus, c.naif_ids, c.naif_frame, c.abcorr,
            interpolated_ephems, c.spherical_harmonics_data, c.frame_PCPF,
            c.factorial_alias, interpolated_transformation,
            c.include_srp, c.k_srp_cannonball, c.idx_sun,
            c.include_drag, c.k_drag, c.omega_atm, c.f_density, c.u)
    else
        ephemerides_backend = _make_ephemerides_backend(
            ephemerides_provider, ephemerides_files, ephemerides_frame_system, c.frame_PCPF)

        return EphemeridesParameters(c.et0, c.DU, c.TU, c.VU,
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