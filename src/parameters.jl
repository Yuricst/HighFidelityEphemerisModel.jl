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
    factorial_alias::Function
    interpolated_transformation::Union{Nothing,InterpolatedTransformation}

    include_srp::Bool
    k_srp_cannonball::Float64
    idx_sun::Int

    include_drag::Bool
    k_drag::Float64
    omega_atm::Vector{Float64}
    f_density::Union{Nothing,Function}

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
    @printf("    include_drag     : %s\n", params.include_drag)
    if params.include_drag
        @printf("    k_drag           : %1.8f\n", params.k_drag)
        @printf("    omega_atm        : [%1.8f, %1.8f, %1.8f]\n", params.omega_atm...)
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
- `include_drag::Bool`: whether to include atmospheric drag terms
- `drag_Cd::Float64`: drag coefficient, dimensionless
- `drag_Am::Float64`: drag area-to-mass ratio in m^2/kg
- `f_density`: callback `(et, r_km) -> rho` returning atmospheric density in kg/m^3
- `omega_atm::Vector{Float64}`: atmospheric rotation rate in rad/s, in the inertial frame
- `nu::Int`: control dimension for vector to be constructed within parameters struct
- `use_canonical_scales::Bool`: whether to use canonical scales for the problem
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
    include_drag::Bool = false,
    drag_Cd::Float64 = 2.2,
    drag_Am::Float64 = 0.01,
    f_density::Union{Nothing,Function} = nothing,
    omega_atm::Vector{Float64} = [0.0, 0.0, 7.2921159e-5],
    nu::Int = 4,
    use_canonical_scales::Bool = true,
)
    # check to see if we use canonical scales
    if use_canonical_scales
        VU = sqrt(GMs[1]/DU)
        TU = DU/VU
        mus = GMs / GMs[1]          # scaled GM's
    else
        DU, TU, VU = 1.0, 1.0, 1.0  # overwrite canonical scales to 1
        mus = GMs                   # unscaled GM's
    end

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

    # Spherical-harmonic acceleration evaluates Legendre terms up to degree nmax + 1,
    # whose summation uses factorial(2 * (nmax + 1)). Int factorial is safe only
    # through factorial(20), so switch before nmax reaches 10.
    factorial_alias = nmax <= 9 ? factorial : factorial_safe
    if !isnothing(filepath_spherical_harmonics)
        spherical_harmonics_data = load_spherical_harmonics(
            filepath_spherical_harmonics, nmax, true, factorial_alias
        )
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

    # drag parameters
    if include_drag
        if isnothing(f_density)
            @error "f_density must be provided when drag is included"
        end
        k_drag = get_drag_coefficient(DU, TU, VU, drag_Cd, drag_Am)
    else
        k_drag = 0.0
    end

    return HighFidelityEphemerisModelParameters(
        et0, DU, TU, VU,
        GMs, mus, naif_ids, naif_frame, abcorr,
        interpolated_ephems,
        spherical_harmonics_data,
        frame_PCPF,
        factorial_alias,
        interpolated_transformation,
        include_srp,
        k_srp_cannonball,
        idx_sun,
        include_drag,
        k_drag,
        omega_atm,
        f_density,
        f_jacobian, Rs, zeros(3),
        nothing,        # adtype, defaults to nothing
        nothing,        # jacobian_cache, defaults to nothing
        zeros(nu),
    )
end