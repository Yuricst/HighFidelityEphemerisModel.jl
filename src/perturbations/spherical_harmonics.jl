"""Perturbations due to spherical harmonics"""


function cart2sph(rvec::Vector)
    x,y,z = rvec
    lmb = atan(y,x)
    phi = atan(z,sqrt(x^2 + y^2))
    r = sqrt(x^2 + y^2 + z^2)
    return lmb, phi, r
end


"""
Compute Legendre function of degree n order m

# Arguments
- `n::Int`: degree (must be >= m)
- `m::Int`: order 
- `t::Real`: argument (cosine of latitude)

# Returns
- `Pnm::Real`: Legendre function value
"""
function legendre(n::Int, m::Int, t::Real, factorial_alias::Function = factorial_safe)
    @assert n >= m "Require n >= m!"
    
    # Get r
    if mod(n - m, 2) == 0
        r = div(n - m, 2)
    else
        r = div(n - m - 1, 2)
    end
    
    # Compute sum term
    sum_term = 0.0
    for k = 0:r
        denom = factorial_alias(k) * factorial_alias(n - k) * factorial_alias(n - m - 2*k)
        sum_term += (-1)^k * factorial_alias(2*n - 2*k) / denom * t^(n - m - 2*k)
    end
    
    # Compute Pnm
    Pnm = 2.0^(-n) * (1 - t^2)^(m/2) * sum_term
    return Pnm
end
    


"""
Get multipliers into gravity potential
c.f. Montenbruck & Gill pg.66 eqn (3.27)

# Arguments
- `phi::Real`: latitude
- `lambda::Real`: longitude
- `R::Real`: radius of Earth
- `r::Real`: radius of spacecraft
- `n::Int`: degree
- `m::Int`: order

# Returns
- `Vnm::Real`: Vnm component
- `Wnm::Real`: Wnm component
"""
function get_VWnm(phi::Real, lambda::Real, R::Real, r::Real, n::Int, m::Int, factorial_alias::Function = factorial_safe)
    Pnm = legendre(n,m,sin(phi), factorial_alias)
    Vnm = (R/r)^(n+1) * Pnm * cos(m*lambda)
    Wnm = (R/r)^(n+1) * Pnm * sin(m*lambda)
    return Vnm, Wnm
end


"""
Get acceleration due to (n,m) potential in planet-centered planet-fixed frame
c.f. Montenbruck & Gill pg.68 eqn (3.33)

# Arguments
- `phi::Real`: latitude
- `lambda::Real`: longitude
- `Cnm_dict::Dict`: dictionary of Cnm coefficients (de-normalized)
- `Snm_dict::Dict`: dictionary of Snm coefficients (de-normalized)
- `GM::Real`: gravitational parameter
- `R::Real`: reference radius
- `r::Real`: radius of spacecraft
- `n::Int`: degree
- `m::Int`: order

# Returns
- `accel_PCPF::Vector{Real}`: acceleration in planet-centered planet-fixed frame
"""
function spherical_harmonics_nm_accel_PCPF(
    phi::Real, lambda::Real, r::Real, 
    Cnm_dict::Dict, Snm_dict::Dict,
    GM::Real, R::Real, n::Int, m::Int,
    factorial_alias::Function = factorial_safe
)
    # Extract coefficients
    Cnm = Cnm_dict[n,m]
    Snm = Snm_dict[n,m]
    
    V_n1_m, W_n1_m = get_VWnm(phi, lambda, R, r, n+1, m, factorial_alias)
    
    if m == 0
        V_n1_1, W_n1_1 = get_VWnm(phi, lambda, R, r, n+1, 1, factorial_alias)
        Cn0 = Cnm_dict[n,0]
        d2x_nm = GM/R^2 * (-Cn0 * V_n1_1)
        d2y_nm = GM/R^2 * (-Cn0 * W_n1_1)
    else
        V_n1_m1, W_n1_m1 = get_VWnm(phi, lambda, R, r, n+1, m+1, factorial_alias)
        V_n1_m_1, W_n1_m_1 = get_VWnm(phi, lambda, R, r, n+1, m-1, factorial_alias)
        d2x_nm = GM/(2*R^2) * (
            (-Cnm * V_n1_m1 - Snm * W_n1_m1) + 
            factorial_alias(n-m+2)/factorial_alias(n-m) * (Cnm * V_n1_m_1 + Snm * W_n1_m_1)
        )
        d2y_nm = GM/(2*R^2) * (
            (-Cnm * W_n1_m1 + Snm * V_n1_m1) + 
            factorial_alias(n-m+2)/factorial_alias(n-m) * (-Cnm * W_n1_m_1 + Snm * V_n1_m_1)
        )
    end
    
    d2z_nm = GM/R^2 * ((n-m+1) * (-Cnm * V_n1_m - Snm * W_n1_m))
    
    return [d2x_nm, d2y_nm, d2z_nm]
end


"""
Get acceleration due to potential up to degree nmax in planet-centered planet-fixed frame

# Arguments
- `phi::Real`: latitude
- `lambda::Real`: longitude
- `CS::Matrix`: coefficient matrix
- `GM::Real`: gravitational parameter
- `R::Real`: reference radius
- `r::Real`: radius of spacecraft
- `nmax::Int`: maximum degree

# Returns
- `accel_PCPF::Vector{Real}`: acceleration in planet-centered planet-fixed frame
"""
function spherical_harmonics_accel_PCPF(
    rvec_PCPF::Vector,
    Cnm_dict::Dict, Snm_dict::Dict,
    GM::Real, R::Real, nmax::Int,
    factorial_alias::Function = factorial_safe
)
    lmb, phi, r = cart2sph(rvec_PCPF)
    accel_PCPF = zeros(3)
    for n = 2:nmax
        for m = 0:n
            accel_PCPF += spherical_harmonics_nm_accel_PCPF(
                phi, lmb, r, Cnm_dict, Snm_dict, GM, R, n, m, factorial_alias
            )
        end
    end
    return accel_PCPF
end


function spherical_harmonics_accel(
    T_inr2pcpf::Matrix{Float64},
    rvec_integrator::Vector,
    Cnm_dict::Dict,
    Snm_dict::Dict,
    GM::Real,
    R::Real,
    nmax::Int,
    factorial_alias::Function = factorial_safe
)
    rvec_PCPF = T_inr2pcpf * rvec_integrator
    return transpose(T_inr2pcpf) * spherical_harmonics_accel_PCPF(
        rvec_PCPF, Cnm_dict, Snm_dict, GM, R, nmax, factorial_alias
    )
end


function load_spherical_harmonics(
    filepath::String,
    nmax::Int,
    denormalize::Bool,
    factorial_alias::Function = factorial_safe,
)
    spherical_harmonics_data = Dict()
    spherical_harmonics_data["nmax"] = nmax
    spherical_harmonics_data["Cnm"] = Dict{Tuple{Int,Int},Float64}()
    spherical_harmonics_data["Snm"] = Dict{Tuple{Int,Int},Float64}()
    open(filepath) do file
        # read first line
        header = readline(file)
        # Split header by commas and parse each value as Float64
        header_values = parse.(Float64, split(header, ","))
        spherical_harmonics_data["REFERENCE RADIUS"] = header_values[1]
        spherical_harmonics_data["GM"] = header_values[2]
        spherical_harmonics_data["NORMALIZATION"] = Int(header_values[6])
        spherical_harmonics_data["REFERENCE LONGITUDE"] = header_values[7]
        spherical_harmonics_data["REFERENCE LATITUDE"] = header_values[8]
        
        # read remaining lines
        for (idx,line) in enumerate(eachline(file))
            # parse line into values
            values = split(line, ",")
            n = parse(Int, strip(values[1]))
            m = parse(Int, strip(values[2]))
            if n < 2
                continue
            end
            C = parse(Float64, strip(values[3]))
            S = parse(Float64, strip(values[4]))
            σC = parse(Float64, strip(values[5]))
            σS = parse(Float64, strip(values[6]))
    
            if denormalize == false
                spherical_harmonics_data["Cnm"][n,m] = C
                spherical_harmonics_data["Snm"][n,m] = S
            elseif spherical_harmonics_data["NORMALIZATION"] == 1
                if m == 0
                    k = 1
                else
                    k = 2
                end

                # undo normalization, c.f. Montenbruck & Gill pg. 58 eqn (3.13)
                spherical_harmonics_data["Cnm"][n,m] = C / sqrt(factorial_alias(n+m)/(k*(2*n+1)*factorial_alias(n-m)));
                spherical_harmonics_data["Snm"][n,m] = S / sqrt(factorial_alias(n+m)/(k*(2*n+1)*factorial_alias(n-m)));
            else
                error("Normalization not supported")
            end
    
            if n == nmax + 1
                break
            end
        end
    end
    return spherical_harmonics_data
end