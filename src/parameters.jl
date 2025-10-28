"""Parameters struct"""


mutable struct HighFidelityEphemerisModelParameters
    et0::Float64
    DU::Real
    TU::Real
    VU::Real

    GMs::Vector{Float64}
    mus::Vector{Float64}
    naif_ids::Vector{String}
    naif_frame::String
    abcorr::String
    interpolated_ephems::Union{Nothing,Vector{InterpolatedEphemeris}}
    spherical_harmonics_data::Union{Nothing,Dict}
    frame_PCPF::Union{Nothing,String}
    interpolated_transformation::Union{Nothing,InterpolatedTransformation}

    include_srp::Bool
    k_srp_cannonball::Float64
    idx_sun::Int

    f_jacobian::Union{Nothing,Function}
    Rs::Vector{Float64}
    R_sun::Vector{Float64}

    adtype
    jacobian_cache

    u::Vector
end


function Base.show(io::IO, params::HighFidelityEphemerisModelParameters)
    println("HighFidelityEphemerisModelParameters struct")
    @printf("    et0              : %s (et = %1.8f)\n", et2utc(params.et0, "ISOC", 3), params.et0)
    @printf("    DU               : %1.8f\n", params.DU)
    @printf("    TU               : %1.8f\n", params.TU)
    @printf("    VU               : %1.8f\n", params.VU)
    @printf("    include_srp      : %s\n", params.include_srp)
    if params.include_srp
        @printf("    k_srp_cannonball : %1.8f\n", params.k_srp_cannonball)
        @printf("    idx_sun          : %d\n", params.idx_sun)
    end
end


"""
Construct HighFidelityEphemerisModelParameters struct.

# Arguments
- `et0::Float64`: reference epoch in seconds past J2000
- `DU::Real`: canonical distance unit
- `GMs::Vector{Float64}`: gravitational constants of the bodies, in km^3/s^2
- `naif_ids::Vector{String}`: NAIF IDs of the bodies
- `naif_frame::String`: inertial frame in which dynamics is integrated
- `abcorr::String`: aberration correction for querying ephemerides of third bodies
- `filepath_spherical_harmonics::Union{Nothing,String}`: path to spherical harmonics data file
- `nmax::Int`: maximum degree of spherical harmonics to be included
- `frame_PCPF::Union{Nothing,String}`: NAIF frame of planet-centered planet-fixed frame
- `get_jacobian_func::Bool`: whether to construct symbolic Jacobian function (only for `Nbody` dynamics)
- `interpolate_ephem_span::Union{Nothing,Vector{Float64}}`: span of epochs to interpolate ephemerides
- `interpolation_time_step::Real`: time step for interpolation
- `include_srp::Bool`: whether to include SRP terms
- `srp_Cr::Float64`: SRP radiation pressure coefficient
- `srp_Am::Float64`: SRP area-to-mass ratio in m^2/kg
- `srp_P0::Float64`: SRP power in W
- `nu::Int`: control dimension for vector to be constructed within parameters struct
"""
function HighFidelityEphemerisModelParameters(
    et0::Float64,
    DU::Real,
    GMs::Vector{Float64},
    naif_ids::Vector{String},
    naif_frame::String = "J2000",
    abcorr::String = "NONE";
    filepath_spherical_harmonics::Union{Nothing,String} = nothing,
    nmax::Int = 4,
    frame_PCPF::Union{Nothing,String} = nothing,
    get_jacobian_func::Bool = true,
    interpolate_ephem_span::Union{Nothing,Vector{Float64}} = nothing,
    interpolation_time_step::Real = 3600.0,
    include_srp::Bool = false,
    srp_Cr::Float64 = 1.15,
    srp_Am::Float64 = 0.002,
    srp_P0::Float64 = 4.56e-6,
    nu::Int = 4,
)
    VU = sqrt(GMs[1]/DU)
    TU = DU/VU
    mus = GMs / GMs[1]         # scaled GM's

    # Jacobian function
    if get_jacobian_func
        if include_srp
            f_jacobian = symbolic_NbodySRP_jacobian(length(GMs))
        else
            f_jacobian = symbolic_Nbody_jacobian(length(GMs))
        end
    else
        f_jacobian = nothing
    end
    Rs = zeros(3 * (length(mus)-1))  # storage for third-body positions

    # initialize interpolated structs
    interpolated_ephems = nothing
    interpolated_transformation = nothing

    if !isnothing(interpolate_ephem_span)
        # interpolate ephemerides of third-bodies
        N_interp = Int(ceil((interpolate_ephem_span[2] - interpolate_ephem_span[1]) / interpolation_time_step))
        ets_interp = range(interpolate_ephem_span[1], interpolate_ephem_span[2], N_interp)
        interpolated_ephems = []
        for ID in naif_ids[2:end]
            rvs_interp = hcat([spkezr(ID, et, naif_frame, abcorr, naif_ids[1])[1] for et in ets_interp]...)
            push!(interpolated_ephems, InterpolatedEphemeris(ID, ets_interp, rvs_interp, false, TU))
        end

        # interpolate transformation matrix from inertial frame to PCPF frame
        if !isnothing(frame_PCPF)
            interpolated_transformation = InterpolatedTransformation(
                ets_interp,
                naif_frame,
                frame_PCPF,
                false,
                TU,
            )
        end
    end

    if !isnothing(filepath_spherical_harmonics)
        spherical_harmonics_data = load_spherical_harmonics(filepath_spherical_harmonics, nmax, true)
        #@assert isnothing(frame_PCPF) == false, "frame_PCPF must be provided when spherical harmonics are used"
    else
        spherical_harmonics_data = nothing
    end

    # SRP parameters
    k_srp_cannonball = get_srp_cannonball_coefficient(DU, TU, srp_Cr, srp_Am, srp_P0)
    if "10" in naif_ids
        idx_sun = findfirst(x -> x == "10", naif_ids)
    elseif include_srp
        @error "NAIF ID \"10\" (Sun) must be provided when SRP is included"
    else
        k_srp_cannonball = 0.0
        idx_sun = 0
    end

    return HighFidelityEphemerisModelParameters(
        et0, DU, TU, VU,
        GMs, mus, naif_ids, naif_frame, abcorr,
        interpolated_ephems,
        spherical_harmonics_data,
        frame_PCPF,
        interpolated_transformation,
        include_srp,
        k_srp_cannonball,
        idx_sun,
        f_jacobian, Rs, zeros(3),
        nothing,        # adtype, defaults to nothing
        nothing,        # jacobian_cache, defaults to nothing
        zeros(nu),
    )
end