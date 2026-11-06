"""Perturbations due to spherical harmonics"""


function cart2sph(rvec::AbstractVector)
    x,y,z = rvec
    lmb = atan(y,x)
    phi = atan(z,sqrt(x^2 + y^2))
    r = sqrt(x^2 + y^2 + z^2)
    return lmb, phi, r
end


"""
Legacy helper (not used by the equations of motion, which use the normalized recursion of
[`spherical_harmonics_accel_normalized`](@ref)).

Compute the (unnormalized) associated Legendre function of degree n order m,
without the Condon-Shortley phase (c.f. Montenbruck & Gill pg.57 eqn (3.11)).

Evaluated with the standard three-term recurrence in degree,

    P_mm      = (2m-1)!! (1 - t^2)^(m/2)
    P_{m+1,m} = (2m+1) t P_mm
    P_nm      = ((2n-1) t P_{n-1,m} - (n+m-1) P_{n-2,m}) / (n-m)

This avoids the explicit-sum formula, which overflowed `Int64` for n >= 13 and lost
precision through cancellation at high degree. Relative to max |P_nm| over t, the
recurrence is accurate to ~1e-13 up to degree 30 (worst near the poles, where
`1 - t^2` itself carries the rounding of t = sin φ). Being unnormalized, the values
exceed `Float64` range for orders m >~ 150 (and the denormalized coefficients become
subnormal), so this formulation is limited to fields of degree well below that.

# Arguments
- `n::Int`: degree (must be >= m)
- `m::Int`: order (must be >= 0)
- `t::Real`: argument (sine of latitude)
- `factorial_alias::Function`: unused; retained for backward compatibility

# Returns
- `Pnm::Real`: Legendre function value
"""
function legendre(n::Int, m::Int, t::Real, factorial_alias::Function = factorial_safe)
    @assert 0 <= m <= n "Require 0 <= m <= n!"

    # P_mm = (2m-1)!! (1 - t^2)^(m/2); even powers are kept polynomial (no sqrt)
    # so that derivatives (e.g. ForwardDiff) stay finite at t = ±1 whenever possible
    one_minus_t2 = (1 - t) * (1 + t)    # more accurate than 1 - t^2 for |t| near 1
    P_mm = one(t)
    for i = 1:m
        P_mm *= 2i - 1                  # (2m-1)!!
    end
    for _ = 1:(m ÷ 2)
        P_mm *= one_minus_t2            # (1 - t^2)^(m ÷ 2), by repeated product
    end
    if isodd(m)
        P_mm *= sqrt(one_minus_t2)
    end
    n == m && return P_mm

    # upward recurrence in degree
    P_prev = P_mm                       # P_{l-2,m}
    P_curr = (2m + 1) * t * P_mm        # P_{l-1,m}
    for l = m+2:n
        P_prev, P_curr = P_curr, ((2l - 1) * t * P_curr - (l + m - 1) * P_prev) / (l - m)
    end
    return P_curr
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

Legacy single-term evaluation with un-normalized functions (each call recomputes the
Legendre functions it needs). Retained for reference/testing only; it is accurate for
low degree but is not used by the equations of motion, which call
[`spherical_harmonics_accel_normalized`](@ref) through [`spherical_harmonics_accel`](@ref).

# Arguments
- `phi::Real`: latitude
- `lambda::Real`: longitude
- `Cnm_dict::AbstractDict`: dictionary of Cnm coefficients (de-normalized)
- `Snm_dict::AbstractDict`: dictionary of Snm coefficients (de-normalized)
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
    Cnm_dict::AbstractDict, Snm_dict::AbstractDict,
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
        factorial_ratio = (n-m+2) * (n-m+1)     # (n-m+2)! / (n-m)!, exact without factorials
        d2x_nm = GM/(2*R^2) * (
            (-Cnm * V_n1_m1 - Snm * W_n1_m1) +
            factorial_ratio * (Cnm * V_n1_m_1 + Snm * W_n1_m_1)
        )
        d2y_nm = GM/(2*R^2) * (
            (-Cnm * W_n1_m1 + Snm * V_n1_m1) +
            factorial_ratio * (-Cnm * W_n1_m_1 + Snm * V_n1_m_1)
        )
    end
    
    d2z_nm = GM/R^2 * ((n-m+1) * (-Cnm * V_n1_m - Snm * W_n1_m))
    
    return [d2x_nm, d2y_nm, d2z_nm]
end


# ------------------------------------------------------------------------------------------
# Fully-normalized gravity field and Cunningham V/W recursion
# ------------------------------------------------------------------------------------------

"""
    SphericalHarmonicsField(Cbar, Sbar)

Fully-normalized spherical-harmonics gravity coefficients up to degree
`nmax = size(Cbar, 1) - 1`, together with precomputed recursion coefficients for the
normalized Cunningham V/W recursion.

`Cbar[n+1, m+1]`, `Sbar[n+1, m+1]` hold the fully-normalized coefficients C̄ₙₘ, S̄ₙₘ
(geodesy normalization, no Condon-Shortley phase), related to the un-normalized ones by
`Cₙₘ = Nₙₘ C̄ₙₘ` with `Nₙₘ = sqrt((2 - δₘ₀)(2n + 1)(n - m)!/(n + m)!)`
(Montenbruck & Gill, *Satellite Orbits*, Eq. (3.13)). Entries with `m > n` and degrees
0 and 1 are ignored by the acceleration.

The coefficient element type `T` is `Float64` for loaded data; other `Real` types
(e.g. ForwardDiff duals, for sensitivities with respect to the coefficients) are
supported. The recursion tables are always `Float64`. The field can be shared between
threads as long as its coefficients are not modified concurrently.
"""
struct SphericalHarmonicsField{T<:Real}
    nmax::Int
    Cbar::Matrix{T}         # [n+1, m+1], 0 <= m <= n <= nmax
    Sbar::Matrix{T}
    # recursion coefficients up to degree nmax + 1 (indices [n+1, m+1] / [m+1])
    A::Matrix{Float64}      # V̄ₙₘ = Aₙₘ z̃ V̄ₙ₋₁,ₘ - Bₙₘ ρ̃ V̄ₙ₋₂,ₘ
    B::Matrix{Float64}
    D::Vector{Float64}      # V̄ₘₘ = Dₘ (x̃ V̄ₘ₋₁,ₘ₋₁ - ỹ W̄ₘ₋₁,ₘ₋₁)
    # acceleration factors for degree n <= nmax (indices [n+1, m+1])
    Fz::Matrix{Float64}     # multiplies V̄ₙ₊₁,ₘ   in z̈
    Fp::Matrix{Float64}     # multiplies V̄ₙ₊₁,ₘ₊₁ in ẍ, ÿ
    Fm::Matrix{Float64}     # multiplies V̄ₙ₊₁,ₘ₋₁ in ẍ, ÿ (m >= 1)
end


function SphericalHarmonicsField(Cbar::AbstractMatrix{<:Real}, Sbar::AbstractMatrix{<:Real})
    size(Cbar) == size(Sbar) || throw(DimensionMismatch("Cbar and Sbar must have the same size"))
    size(Cbar, 1) == size(Cbar, 2) || throw(DimensionMismatch("Cbar and Sbar must be square"))
    nmax = size(Cbar, 1) - 1
    L = nmax + 2

    A = zeros(L, L)
    B = zeros(L, L)
    D = zeros(L)
    for m = 0:nmax+1
        if m >= 1
            # (2m-1) Nₘₘ/Nₘ₋₁,ₘ₋₁ = sqrt(kₘ/kₘ₋₁ (2m+1)/(2m)), k₀ = 1, kₘ = 2 (m >= 1)
            D[m+1] = sqrt((m == 1 ? 2.0 : 1.0) * (2m + 1) / (2m))
        end
        for n = m+1:nmax+1
            A[n+1, m+1] = sqrt((2n - 1) * (2n + 1) / ((n - m) * (n + m)))
            if n >= m + 2
                B[n+1, m+1] = sqrt((2n + 1) * (n + m - 1) * (n - m - 1) / ((2n - 3) * (n + m) * (n - m)))
            end
        end
    end

    Fz = zeros(nmax + 1, nmax + 1)
    Fp = zeros(nmax + 1, nmax + 1)
    Fm = zeros(nmax + 1, nmax + 1)
    for n = 0:nmax, m = 0:n
        c = (2n + 1) / (2n + 3)
        # (n-m+1) Nₙₘ/Nₙ₊₁,ₘ
        Fz[n+1, m+1] = sqrt(c * (n + m + 1) * (n - m + 1))
        # Nₙₘ/Nₙ₊₁,ₘ₊₁ (kₘ/kₘ₊₁ = 1/2 for m = 0)
        Fp[n+1, m+1] = sqrt((m == 0 ? 0.5 : 1.0) * c * (n + m + 1) * (n + m + 2))
        if m >= 1
            # (n-m+2)!/(n-m)! Nₙₘ/Nₙ₊₁,ₘ₋₁ (kₘ/kₘ₋₁ = 2 for m = 1)
            Fm[n+1, m+1] = sqrt((m == 1 ? 2.0 : 1.0) * c * (n - m + 2) * (n - m + 1))
        end
    end

    T = float(promote_type(eltype(Cbar), eltype(Sbar)))
    return SphericalHarmonicsField{T}(nmax, Matrix{T}(Cbar), Matrix{T}(Sbar), A, B, D, Fz, Fp, Fm)
end


"""
    SphericalHarmonicsCoefficients <: AbstractDict{Tuple{Int,Int},Float64}

Dictionary view `(n, m) => coefficient` of one coefficient set (C or S) of a
[`SphericalHarmonicsField`](@ref), for degrees `2 <= n <= nmax`, `0 <= m <= n`.

This is the type stored in `load_spherical_harmonics(...)["Cnm"]` / `["Snm"]`. Indexing
(`Cnm[n, m]`) returns un-normalized coefficients when the data were loaded with
`denormalize = true` (the historical behaviour; values computed on access and subject to
`Float64` underflow at high degree), and the fully-normalized ones otherwise. The
acceleration routines never use these values; they use the normalized matrices of the
underlying field directly.

Assigning an existing key (`Cnm[2, 0] = value` or `Cnm[(2, 0)] = value`, in the same
normalization as indexing returns) writes through to the underlying field, so every
object sharing that field (including the `"Cnm"`/`"Snm"` partner view and parameter
copies) sees the change. Keys outside `2 <= n <= nmax`, `0 <= m <= n` cannot be added,
and values must be convertible to `Float64`. To use other coefficient types (e.g.
ForwardDiff duals) or a different set of keys, replace `sh["Cnm"]`/`sh["Snm"]` by an
ordinary `Dict` of un-normalized coefficients; the equations of motion then use the
(slower) dictionary path.
"""
struct SphericalHarmonicsCoefficients <: AbstractDict{Tuple{Int,Int},Float64}
    field::SphericalHarmonicsField{Float64}
    sine::Bool
    denormalized::Bool
end


@inline _sh_has(c::SphericalHarmonicsCoefficients, n::Integer, m::Integer) =
    2 <= n <= c.field.nmax && 0 <= m <= n

function _sh_value(c::SphericalHarmonicsCoefficients, n::Int, m::Int)
    v = c.sine ? c.field.Sbar[n+1, m+1] : c.field.Cbar[n+1, m+1]
    c.denormalized || return v
    # undo normalization, c.f. Montenbruck & Gill pg. 58 eqn (3.13)
    return Float64(v / spherical_harmonics_normalization_scale(n, m, m == 0 ? 1 : 2))
end

Base.length(c::SphericalHarmonicsCoefficients) = max(0, (c.field.nmax + 1) * (c.field.nmax + 2) ÷ 2 - 3)
Base.haskey(c::SphericalHarmonicsCoefficients, key::Tuple{Integer,Integer}) = _sh_has(c, key...)
Base.haskey(c::SphericalHarmonicsCoefficients, key) = false

function Base.get(c::SphericalHarmonicsCoefficients, key::Tuple{Integer,Integer}, default)
    return _sh_has(c, key...) ? _sh_value(c, Int(key[1]), Int(key[2])) : default
end
Base.get(c::SphericalHarmonicsCoefficients, key, default) = default

function Base.getindex(c::SphericalHarmonicsCoefficients, key::Tuple{Integer,Integer})
    _sh_has(c, key...) || throw(KeyError(key))
    return _sh_value(c, Int(key[1]), Int(key[2]))
end
Base.getindex(c::SphericalHarmonicsCoefficients, n::Integer, m::Integer) = c[(n, m)]

function Base.setindex!(c::SphericalHarmonicsCoefficients, value, key::Tuple{Integer,Integer})
    n, m = Int(key[1]), Int(key[2])
    _sh_has(c, n, m) || throw(KeyError(key))
    v = try
        convert(Float64, value)
    catch
        throw(ArgumentError(
            "SphericalHarmonicsCoefficients store Float64 coefficients; cannot assign a value of " *
            "type $(typeof(value)). To use e.g. ForwardDiff duals, replace sh[\"Cnm\"] / sh[\"Snm\"] " *
            "by ordinary Dicts of un-normalized coefficients."
        ))
    end
    if c.denormalized
        # store the normalized value, c.f. Montenbruck & Gill pg. 58 eqn (3.13)
        v = Float64(v * spherical_harmonics_normalization_scale(n, m, m == 0 ? 1 : 2))
    end
    M = c.sine ? c.field.Sbar : c.field.Cbar
    M[n+1, m+1] = v
    return c
end
Base.setindex!(c::SphericalHarmonicsCoefficients, value, n::Integer, m::Integer) = setindex!(c, value, (n, m))


function Base.iterate(c::SphericalHarmonicsCoefficients, state = (2, 0))
    n, m = state
    n > c.field.nmax && return nothing
    next = m < n ? (n, m + 1) : (n + 1, 0)
    return ((n, m) => _sh_value(c, n, m)), next
end


"""
    _sh_fill_column!(V, W, m, nmax_deg, zt, rho, A, B)

Fill degrees `m+1 .. nmax_deg` of column (order) `m` of the normalized V̄/W̄ arrays
(index `n+1`), given the sectoral value at index `m+1`.
"""
@inline function _sh_fill_column!(V, W, m::Int, nmax_deg::Int, zt, rho, A, B)
    m + 1 > nmax_deg && return nothing
    a = A[m+2, m+1] * zt
    V[m+2] = a * V[m+1]
    W[m+2] = a * W[m+1]
    @inbounds for n = m+2:nmax_deg
        a = A[n+1, m+1] * zt
        b = B[n+1, m+1] * rho
        V[n+1] = a * V[n] - b * V[n-1]
        W[n+1] = a * W[n] - b * W[n-1]
    end
    return nothing
end


"""
    spherical_harmonics_accel_normalized(x, y, z, field, GM, R, nmax)

Body-fixed acceleration `(ax, ay, az)` of the spherical-harmonics terms of degree
`2..nmax` at the body-fixed position `(x, y, z)` (km in, km/s² out for GM in km³/s²).

Implements Montenbruck & Gill, *Satellite Orbits*, §3.2.4-3.2.5, Eqs. (3.29)-(3.33),
rewritten for fully-normalized quantities V̄ₙₘ = Nₙₘ Vₙₘ, W̄ₙₘ = Nₙₘ Wₙₘ:

    V̄₀₀ = R/r,  V̄ₘₘ = Dₘ (x̃ V̄ₘ₋₁,ₘ₋₁ - ỹ W̄ₘ₋₁,ₘ₋₁),  W̄ₘₘ = Dₘ (x̃ W̄ₘ₋₁,ₘ₋₁ + ỹ V̄ₘ₋₁,ₘ₋₁)
    V̄ₙₘ = Aₙₘ z̃ V̄ₙ₋₁,ₘ - Bₙₘ ρ̃ V̄ₙ₋₂,ₘ        (same for W̄)
    Aₙₘ = sqrt((2n-1)(2n+1)/((n-m)(n+m))),  Bₙₘ = sqrt((2n+1)(n+m-1)(n-m-1)/((2n-3)(n+m)(n-m)))

with x̃ = xR/r², ỹ = yR/r², z̃ = zR/r², ρ̃ = R²/r² (the standard forward-column
recursion of fully-normalized functions, cf. Holmes & Featherstone, J. Geodesy 76, 2002).
The (n-m+2)!/(n-m)! and (n-m+1) factors of Eq. (3.33) are combined with the
normalization ratios into the precomputed tables `Fp`, `Fm`, `Fz`, so no factorials,
`BigInt`s or un-normalized coefficients are involved. The recursion is Cartesian
(no singularity at the poles) and numerically stable to high degree; only three columns
of V̄/W̄ are kept (O(nmax) memory, one pass, O(nmax²) operations per call).

Generic over `Real` (ForwardDiff duals propagate through). At very high degree
(≳ 1500-2000) sectoral terms near the poles underflow, as with any non-scaled
forward-column recursion (Holmes & Featherstone 2002); this is not handled.
"""
function spherical_harmonics_accel_normalized(
    x::Real, y::Real, z::Real,
    field::SphericalHarmonicsField,
    GM::Real, R::Real, nmax::Int,
)
    nmax <= field.nmax || throw(ArgumentError(
        "requested nmax = $nmax exceeds the degree of the loaded field ($(field.nmax))"
    ))
    TT = float(promote_type(typeof(x), typeof(y), typeof(z), typeof(R), typeof(GM), eltype(field.Cbar)))
    nmax < 2 && return zero(TT), zero(TT), zero(TT)

    r2 = x * x + y * y + z * z
    inv_r2 = 1 / r2
    xt = x * R * inv_r2
    yt = y * R * inv_r2
    zt = z * R * inv_r2
    rho = R * R * inv_r2

    A, B, D = field.A, field.B, field.D
    Cbar, Sbar = field.Cbar, field.Sbar
    Fz, Fp, Fm = field.Fz, field.Fp, field.Fm

    L = nmax + 2                    # degrees 0 .. nmax+1
    nmax_deg = nmax + 1
    Vp = zeros(TT, L); Wp = zeros(TT, L)    # column m-1
    Vc = zeros(TT, L); Wc = zeros(TT, L)    # column m
    Vn = zeros(TT, L); Wn = zeros(TT, L)    # column m+1

    # column 0
    Vc[1] = R / sqrt(r2)
    _sh_fill_column!(Vc, Wc, 0, nmax_deg, zt, rho, A, B)
    # column 1
    Vn[2] = D[2] * (xt * Vc[1] - yt * Wc[1])
    Wn[2] = D[2] * (xt * Wc[1] + yt * Vc[1])
    _sh_fill_column!(Vn, Wn, 1, nmax_deg, zt, rho, A, B)

    ax = zero(TT); ay = zero(TT); az = zero(TT)
    for m = 0:nmax
        @inbounds for n = max(m, 2):nmax
            C = Cbar[n+1, m+1]
            S = Sbar[n+1, m+1]
            i = n + 2               # index of degree n+1
            if m == 0
                fp = Fp[n+1, 1]
                ax -= fp * C * Vn[i]
                ay -= fp * C * Wn[i]
                az -= Fz[n+1, 1] * (C * Vc[i] + S * Wc[i])
            else
                fp = Fp[n+1, m+1]
                fm = Fm[n+1, m+1]
                ax += (fp * (-C * Vn[i] - S * Wn[i]) + fm * (C * Vp[i] + S * Wp[i])) / 2
                ay += (fp * (-C * Wn[i] + S * Vn[i]) + fm * (-C * Wp[i] + S * Vp[i])) / 2
                az -= Fz[n+1, m+1] * (C * Vc[i] + S * Wc[i])
            end
        end
        m == nmax && break

        # shift columns (m-1, m, m+1) -> (m, m+1, m+2), recycling the old m-1 buffer
        Vp, Vc, Vn = Vc, Vn, Vp
        Wp, Wc, Wn = Wc, Wn, Wp
        k = m + 2
        Vn[k+1] = D[k+1] * (xt * Vc[k] - yt * Wc[k])
        Wn[k+1] = D[k+1] * (xt * Wc[k] + yt * Vc[k])
        _sh_fill_column!(Vn, Wn, k, nmax_deg, zt, rho, A, B)
    end

    s = GM / (R * R)
    return s * ax, s * ay, s * az
end


"""
    _sh_field_from_dicts(Cnm_dict, Snm_dict, nmax)

Build a [`SphericalHarmonicsField`](@ref) from dictionaries of **un-normalized**
coefficients `(n, m) => value` (missing entries are treated as zero). Used only for the
backward-compatible plain-`Dict` methods; the scale factors are formed in `Float64`, so
this path is limited to degrees ≲ 150.
"""
function _sh_field_from_dicts(Cnm_dict::AbstractDict, Snm_dict::AbstractDict, nmax::Int)
    T = float(promote_type(_sh_coef_type(Cnm_dict), _sh_coef_type(Snm_dict)))
    Cbar = _sh_normalized_matrix(T, Cnm_dict, nmax)
    Sbar = _sh_normalized_matrix(T, Snm_dict, nmax)
    return SphericalHarmonicsField(Cbar, Sbar)
end


# Element type of the coefficients stored in a dictionary (generic, e.g. ForwardDiff duals).
function _sh_coef_type(d::AbstractDict)
    V = valtype(d)
    (isconcretetype(V) && V <: Real) && return V
    T = Float64
    for v in values(d)
        T = promote_type(T, typeof(v))
    end
    return T
end
_sh_coef_type(::SphericalHarmonicsCoefficients) = Float64


# Fully-normalized (nmax+1) x (nmax+1) coefficient matrix from un-normalized `(n, m)`
# dictionary entries (missing entries are zero).
function _sh_normalized_matrix(::Type{T}, d::AbstractDict, nmax::Int) where {T}
    M = zeros(T, nmax + 1, nmax + 1)
    for n = 2:nmax
        scale = 1 / sqrt(2n + 1)        # sqrt((n+m)!/((n-m)! k (2n+1))) at m = 0
        for m = 0:n
            if m >= 1
                scale *= sqrt((n + m) * (n - m + 1) / (m == 1 ? 2.0 : 1.0))
            end
            M[n+1, m+1] = get(d, (n, m), zero(T)) * scale
        end
    end
    return M
end

# A coefficient view already holds normalized values: copy them (independent of the
# view's `denormalized` flag, which only affects what indexing returns).
function _sh_normalized_matrix(::Type{T}, c::SphericalHarmonicsCoefficients, nmax::Int) where {T}
    nmax <= c.field.nmax || throw(ArgumentError(
        "requested nmax = $nmax exceeds the degree of the loaded field ($(c.field.nmax))"
    ))
    src = c.sine ? c.field.Sbar : c.field.Cbar
    M = zeros(T, nmax + 1, nmax + 1)
    for n = 2:nmax, m = 0:n
        M[n+1, m+1] = src[n+1, m+1]
    end
    return M
end


function _sh_field(Cnm, Snm, nmax::Int)
    # fast path: the C and S views of one loaded field (no copy)
    _sh_views_of_same_field(Cnm, Snm) && return Cnm.field
    # otherwise (plain dictionaries, mixed view/dictionary, ...) build a field; views
    # contribute their normalized values, dictionaries are un-normalized
    return _sh_field_from_dicts(Cnm, Snm, nmax)
end


@inline _sh_views_of_same_field(Cnm, Snm) =
    Cnm isa SphericalHarmonicsCoefficients && Snm isa SphericalHarmonicsCoefficients &&
    Cnm.field === Snm.field && !Cnm.sine && Snm.sine


"""
    spherical_harmonics_accel_PCPF(rvec_PCPF, Cnm, Snm, GM, R, nmax, factorial_alias=nothing)

Acceleration due to the spherical-harmonics terms of degree `2..nmax`, in the
planet-centered planet-fixed (PCPF) frame.

# Arguments
- `rvec_PCPF::AbstractVector`: position in the PCPF frame (same length unit as `R`)
- `Cnm`, `Snm`: coefficients, either the [`SphericalHarmonicsCoefficients`](@ref) returned
  by [`load_spherical_harmonics`](@ref) (fast path, normalized coefficients used directly),
  or plain dictionaries `(n, m) => value` of **un-normalized** coefficients
  (backward-compatible path; normalized internally on every call, so slower)
- `GM::Real`: gravitational parameter
- `R::Real`: reference radius
- `nmax::Int`: maximum degree
- `factorial_alias`: ignored; accepted for backward compatibility

Uses the normalized Cunningham recursion of
[`spherical_harmonics_accel_normalized`](@ref). ForwardDiff-compatible.

# Returns
- `accel_PCPF::Vector`: acceleration in the PCPF frame
"""
function spherical_harmonics_accel_PCPF(
    rvec_PCPF::AbstractVector,
    Cnm::AbstractDict, Snm::AbstractDict,
    GM::Real, R::Real, nmax::Int,
    factorial_alias = nothing,
)
    field = _sh_field(Cnm, Snm, nmax)
    ax, ay, az = spherical_harmonics_accel_normalized(
        rvec_PCPF[1], rvec_PCPF[2], rvec_PCPF[3], field, GM, R, nmax,
    )
    return [ax, ay, az]
end


"""
    spherical_harmonics_accel(T_inr2pcpf, rvec_integrator, Cnm, Snm, GM, R, nmax,
        factorial_alias=nothing)

Spherical-harmonics acceleration in the integration (inertial) frame: rotates
`rvec_integrator` to the PCPF frame with `T_inr2pcpf`, evaluates
[`spherical_harmonics_accel_PCPF`](@ref), and rotates the result back with
`transpose(T_inr2pcpf)`. `factorial_alias` is ignored (backward compatibility).
"""
function spherical_harmonics_accel(
    T_inr2pcpf::AbstractMatrix,
    rvec_integrator::AbstractVector,
    Cnm::AbstractDict,
    Snm::AbstractDict,
    GM::Real,
    R::Real,
    nmax::Int,
    factorial_alias = nothing,
)
    field = _sh_field(Cnm, Snm, nmax)
    T = T_inr2pcpf
    r1, r2, r3 = rvec_integrator[1], rvec_integrator[2], rvec_integrator[3]
    x = T[1, 1] * r1 + T[1, 2] * r2 + T[1, 3] * r3
    y = T[2, 1] * r1 + T[2, 2] * r2 + T[2, 3] * r3
    z = T[3, 1] * r1 + T[3, 2] * r2 + T[3, 3] * r3
    ax, ay, az = spherical_harmonics_accel_normalized(x, y, z, field, GM, R, nmax)
    return [
        T[1, 1] * ax + T[2, 1] * ay + T[3, 1] * az,
        T[1, 2] * ax + T[2, 2] * ay + T[3, 2] * az,
        T[1, 3] * ax + T[2, 3] * ay + T[3, 3] * az,
    ]
end


function spherical_harmonics_normalization_scale(n::Int, m::Int, k::Int)
    factorial_ratio = BigFloat(1)
    for i = (n - m + 1):(n + m)
        factorial_ratio *= BigFloat(i)
    end
    return sqrt(factorial_ratio / (BigFloat(k) * BigFloat(2*n + 1)))
end


"""
    load_spherical_harmonics(filepath, nmax, denormalize, factorial_alias=nothing)

Load a gravity field in the PDS SHADR `.tab` format (as in `data/`) up to degree `nmax`.

The first line holds the reference radius, GM, ..., the normalization state (column 6:
`1` = fully normalized, `0` = un-normalized) and the reference longitude/latitude; the
following lines hold `n, m, C, S, σC, σS`. Coefficients are stored internally fully
normalized in a [`SphericalHarmonicsField`](@ref) (un-normalized files are normalized on
load), which is what the acceleration routines use, so any degree available in the file
can be used (no `BigInt`/factorial limits).

# Returns
A `Dict` with keys `"nmax"`, `"REFERENCE RADIUS"`, `"GM"`, `"NORMALIZATION"`,
`"REFERENCE LONGITUDE"`, `"REFERENCE LATITUDE"`, `"FIELD"` (the
[`SphericalHarmonicsField`](@ref)), and `"Cnm"`, `"Snm"`
([`SphericalHarmonicsCoefficients`](@ref) views). Indexing `"Cnm"`/`"Snm"` returns
un-normalized values if `denormalize == true` (historical behaviour) and normalized
values otherwise; this flag does not affect the computed accelerations.
`factorial_alias` is ignored (backward compatibility).
"""
function load_spherical_harmonics(
    filepath::String,
    nmax::Int,
    denormalize::Bool,
    factorial_alias = nothing,
)
    spherical_harmonics_data = Dict{String,Any}()
    spherical_harmonics_data["nmax"] = nmax
    Cbar = zeros(nmax + 1, nmax + 1)
    Sbar = zeros(nmax + 1, nmax + 1)
    max_degree_read = -1
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
        normalization = spherical_harmonics_data["NORMALIZATION"]
        normalization in (0, 1) || error(
            "Normalization state $normalization not supported (expected 0 = un-normalized or 1 = fully normalized)"
        )

        # read remaining lines
        for line in eachline(file)
            isempty(strip(line)) && continue
            values = split(line, ",")
            n = parse(Int, strip(values[1]))
            m = parse(Int, strip(values[2]))
            n > nmax && break
            max_degree_read = max(max_degree_read, n)
            n < 2 && continue
            C = parse(Float64, strip(values[3]))
            S = parse(Float64, strip(values[4]))
            if normalization == 0
                # normalize, c.f. Montenbruck & Gill pg. 58 eqn (3.13)
                scale = Float64(spherical_harmonics_normalization_scale(n, m, m == 0 ? 1 : 2))
                C *= scale
                S *= scale
            end
            Cbar[n+1, m+1] = C
            Sbar[n+1, m+1] = S
        end
    end
    max_degree_read < nmax && error(
        "Gravity field file $filepath only contains coefficients up to degree $max_degree_read < nmax = $nmax"
    )

    field = SphericalHarmonicsField(Cbar, Sbar)
    spherical_harmonics_data["FIELD"] = field
    spherical_harmonics_data["Cnm"] = SphericalHarmonicsCoefficients(field, false, denormalize)
    spherical_harmonics_data["Snm"] = SphericalHarmonicsCoefficients(field, true, denormalize)
    return spherical_harmonics_data
end
