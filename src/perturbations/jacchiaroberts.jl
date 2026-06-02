"""Jacchia-Roberts atmosphere model (GMAT heritage)"""

const _JR_RHO_ZERO = 3.46e-9
const _JR_TZERO = 183.0
const _JR_G_ZERO = 9.80665
const _JR_GAS_CON = 8.31432
const _JR_AVOGADRO = 6.022045e23
const _JR_PI = π

const _JR_CON_C = [-89284375.0, 3542400.0, -52687.5, 340.5, -0.8]
const _JR_CON_L = [0.1031445e5, 0.2341230e1, 0.1579202e-2, -0.1252487e-5, 0.2462708e-9]
const _JR_MZERO = 28.82678
const _JR_M_CON = [
    -435093.363387, 28275.5646391, -765.33466108, 11.043387545,
    -0.08958790995, 0.00038737586, -0.000000697444,
]
const _JR_S_CON = [
    3144902516.672729, -123774885.4832917, 1816141.096520398,
    -11403.31079489267, 24.36498612105595, 0.008957502869707995,
]
const _JR_S_BETA = [-52864482.17910969, -16632.50847336828, -1.308252378125, 0.0, 0.0, 0.0]
const _JR_OMEGA = -0.94585589
const _JR_ZETA_CON = [
    0.1985549e-10, -0.1833490e-14, 0.1711735e-17, -0.1021474e-20,
    0.3727894e-24, -0.7734110e-28, 0.7026942e-32,
]
const _JR_MOL_MASS = [28.0134, 39.948, 4.0026, 31.9988, 15.9994, 1.00797]
const _JR_NUM_DENS = [0.78110, 0.93432e-2, 0.61471e-5, 0.161778, 0.95544e-1]
const _JR_CON_DEN = [
    [0.1093155e2, 0.1186783e-2, -0.1677341e-5, 0.1420228e-8, -0.7139785e-12, 0.1969715e-15, -0.2296182e-19],
    [0.8049405e1, 0.2382822e-2, -0.3391366e-5, 0.2909714e-8, -0.1481702e-11, 0.4127600e-15, -0.4837461e-19],
    [0.7646886e1, -0.4383486e-3, 0.4694319e-6, -0.2894886e-9, 0.9451989e-13, -0.1270838e-16, 0.0],
    [0.9924237e1, 0.1600311e-2, -0.2274761e-5, 0.1938454e-8, -0.9782183e-12, 0.2698450e-15, -0.3131808e-19],
    [0.1097083e2, 0.6118742e-4, -0.1165003e-6, 0.9239354e-10, -0.3490739e-13, 0.5116298e-17, 0.0],
]
const _JR_DAY58_EPOCH_MJD = 36204.0
const _JR_DEFAULT_R_POLAR_KM = 6356.766


mutable struct JacchiaRobertsWorkspace
    t_infinity::Float64
    tx::Float64
    root1::Float64
    root2::Float64
    x_root::Float64
    y_root::Float64
    sum_L::Float64
end

JacchiaRobertsWorkspace() = JacchiaRobertsWorkspace(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


"""
    JacchiaRobertsGeomagneticExposphericParams

Geomagnetic and exospheric inputs for the Jacchia-Roberts atmosphere model.

# Fields
- `tkp`: geomagnetic index Kp (dimensionless)
- `xtemp`: minimum global exospheric temperature at 120 km, in K

# Constructors
    JacchiaRobertsGeomagneticExposphericParams(F107, F107a, Kp)

Build from constant solar and geomagnetic indices (GMAT constant-flux mode).

The exospheric temperature is `379.0 + 3.24 * F107a + 1.3 * (F107 - F107a)` K; `Kp` is
stored as `tkp`.

# Arguments
- `F107`: daily solar flux at 10.7 cm, in SFU (solar flux units)
- `F107a`: 81-day centered average of F10.7, in SFU
- `Kp`: geomagnetic index (dimensionless)
"""
struct JacchiaRobertsGeomagneticExposphericParams
    tkp::Float64
    xtemp::Float64

    function JacchiaRobertsGeomagneticExposphericParams(
        F107::Real, F107a::Real, Kp::Real,
    )
        xtemp = 379.0 + 3.24 * F107a + 1.3 * (F107 - F107a)
        return new(float(Kp), float(xtemp))
    end
end


function _horner(coeffs, x)
    s = coeffs[end]
    for i in (length(coeffs) - 1):-1:1
        s = coeffs[i] + s * x
    end
    return s
end


function _roots!(a::Vector{Float64}, croots::Matrix{Float64}, irl::Int)
    na = length(a)
    ir = 0
    n1 = na - 1
    n2 = n1 - 1
    while ir < irl
        z = [croots[ir + 1, 1], croots[ir + 1, 2]]
        dif = 0.0
        while true
            cb = [a[n1 + 1], 0.0]
            cc = [a[n1 + 1], 0.0]
            for i in 0:n2
                j = n2 - i + 1
                temp = z[1] * cb[1] - z[2] * cb[2] + a[j]
                cb[2] = z[1] * cb[2] + z[2] * cb[1]
                cb[1] = temp
                if j != 1
                    temp = z[1] * cc[1] - z[2] * cc[2] + cb[1]
                    cc[2] = z[1] * cc[2] + z[2] * cc[1] + cb[2]
                    cc[1] = temp
                end
            end
            zs = copy(z)
            denom = cc[1]^2 + cc[2]^2
            z[1] -= (cb[1] * cc[1] + cb[2] * cc[2]) / denom
            z[2] += (cb[1] * cc[2] - cb[2] * cc[1]) / denom
            dif = abs((zs[1] - z[1]) / zs[1])
            if zs[2] != 0.0
                dif += abs((zs[2] - z[2]) / zs[2])
            end
            dif <= 1.0e-14 && break
        end
        croots[ir + 1, 1] = z[1]
        croots[ir + 1, 2] = z[2]
        ir += 1
    end
    return nothing
end


function _deflate_polynomial!(c::Vector{Float64}, root::Float64, c_new::Vector{Float64})
    n = length(c)
    sumv = c[n]
    for i in (n - 1):-1:1
        save = c[i]
        c_new[i] = sumv
        sumv = save + sumv * root
    end
    return nothing
end


function exotherm!(
    ws::JacchiaRobertsWorkspace,
    space_craft::AbstractVector{<:Real},
    sun::AbstractVector{<:Real},
    geo::JacchiaRobertsGeomagneticExposphericParams,
    height::Real,
    sun_dec::Real,
    geo_lat::Real,
    R_polar_km::Real,
)
    sun_denom = hypot(sun[1], sun[2])
    cross_denom = abs(sun[1] * space_craft[2] - sun[2] * space_craft[1])
    cos_denom = hypot(space_craft[1], space_craft[2])
    if cross_denom < 1.0e-15 || cos_denom < 1.0e-15
        error("Jacchia-Roberts exotherm: denominator too close to zero")
    end

    cos_alpha = (sun[1] * space_craft[1] + sun[2] * space_craft[2]) / (sun_denom * cos_denom)
    hour_angle = if cos_alpha >= 1.0 - 1.0e-14
        0.0
    elseif cos_alpha <= -1.0 + 1.0e-14
        ((sun[1] * space_craft[2] - sun[2] * space_craft[1]) / cross_denom) * _JR_PI
    else
        ((sun[1] * space_craft[2] - sun[2] * space_craft[1]) / cross_denom) *
        acos(cos_alpha)
    end

    theta = 0.5 * abs(geo_lat + sun_dec)
    eta = 0.5 * abs(geo_lat - sun_dec)
    tau = hour_angle - 0.64577182325 + 0.10471975512 * sin(hour_angle + 0.75049157836)
    if tau < -_JR_PI
        tau += 2 * _JR_PI
    elseif tau > _JR_PI
        tau -= 2 * _JR_PI
    end
    th22 = sin(theta)^2.2
    t1 = geo.xtemp * (1.0 + 0.3 * (th22 + cos(0.5 * tau)^3.0 * (cos(eta)^2.2 - th22)))
    expkp = exp(geo.tkp)

    ws.t_infinity = if height < 200.0
        t1 + 14.0 * geo.tkp + 0.02 * expkp
    else
        t1 + 28.0 * geo.tkp + 0.03 * expkp
    end

    ws.tx = 371.6678 + 0.0518806 * ws.t_infinity - 294.3505 * exp(-0.00216222 * ws.t_infinity)

    if height < 125.0
        sum_c = _horner(_JR_CON_C, height)
        exotemp = ws.tx + (ws.tx - _JR_TZERO) * sum_c / 1.500625e6
    elseif height > 125.0
        ws.sum_L = _horner(_JR_CON_L, ws.t_infinity)
        exotemp = ws.t_infinity - (ws.t_infinity - ws.tx) * exp(
            -(ws.tx - _JR_TZERO) / (ws.t_infinity - ws.tx) *
            (height - 125.0) / 35.0 * ws.sum_L / (R_polar_km + height),
        )
    else
        exotemp = ws.tx
    end

    if height <= 125.0
        c_star = Vector{Float64}(undef, 5)
        c_star[1] = _JR_CON_C[1] + 1.500625e6 * ws.tx / (ws.tx - _JR_TZERO)
        c_star[2:5] .= _JR_CON_C[2:5]

        aux = zeros(2, 2)
        aux[1, 1] = 125.0
        aux[1, 2] = 0.0
        _roots!(c_star, aux, 1)
        ws.root1 = aux[1, 1]
        _deflate_polynomial!(c_star, ws.root1, c_star)

        aux[1, 1] = 200.0
        aux[1, 2] = 0.0
        _roots!(c_star, aux, 1)
        ws.root2 = aux[1, 1]
        _deflate_polynomial!(c_star, ws.root2, c_star)

        aux[1, 1] = 10.0
        aux[1, 2] = 125.0
        _roots!(c_star, aux, 1)
        ws.x_root = aux[1, 1]
        ws.y_root = abs(aux[1, 2])
    end

    return exotemp
end


function rho_100(
    height::Real,
    temperature::Real,
    ws::JacchiaRobertsWorkspace,
    R_polar_km::Real,
)
    m_poly = _horner(_JR_M_CON, height)
    b = [_JR_S_CON[i] + _JR_S_BETA[i] * ws.tx / (ws.tx - _JR_TZERO) for i in 1:6]

    roots_2 = ws.x_root^2 + ws.y_root^2
    x_star = -2.0 * ws.root1 * ws.root2 * R_polar_km * (
        R_polar_km^2 + 2.0 * R_polar_km * ws.x_root + roots_2
    )
    v = (R_polar_km + ws.root1) * (R_polar_km + ws.root2) * (
        R_polar_km^2 + 2.0 * R_polar_km * ws.x_root + roots_2
    )
    u1 = (ws.root1 - ws.root2) * (ws.root1 + R_polar_km)^2 *
         (ws.root1^2 - 2.0 * ws.root1 * ws.x_root + roots_2)
    u2 = (ws.root1 - ws.root2) * (ws.root2 + R_polar_km)^2 *
         (ws.root2^2 - 2.0 * ws.root2 * ws.x_root + roots_2)
    w1 = ws.root1 * ws.root2 * R_polar_km * (R_polar_km + ws.root1) *
         (R_polar_km + roots_2 / ws.root1)
    w2 = ws.root1 * ws.root2 * R_polar_km * (R_polar_km + ws.root2) *
         (R_polar_km + roots_2 / ws.root2)

    s_poly = _horner(b, ws.root1)
    p2 = s_poly / u1
    s_poly = _horner(b, ws.root2)
    p3 = -s_poly / u2
    s_poly = b[6]
    for i in 5:-1:1
        s_poly = -s_poly * R_polar_km + b[i]
    end
    p5 = s_poly / v
    p4 = (
        b[1] - ws.root1 * ws.root2 * R_polar_km^2 * (
            b[5] + b[6] * (2.0 * ws.x_root + ws.root1 + ws.root2 - R_polar_km)
        ) + w1 * p2 + w2 * p3 - ws.root1 * ws.root2 * b[6] * R_polar_km * roots_2 +
        ws.root1 * ws.root2 * (R_polar_km^2 - roots_2) * p5
    ) / x_star
    p1 = b[6] - 2 * p4 - p3 - p2
    p6 = b[5] + b[6] * (2.0 * ws.x_root + ws.root1 + ws.root2 - R_polar_km) - p5 -
         2.0 * (ws.x_root + R_polar_km) * p4 - (ws.root2 + R_polar_km) * p3 -
         (ws.root1 + R_polar_km) * p2

    log_f1 = p1 * log((height + R_polar_km) / (90.0 + R_polar_km)) +
             p2 * log((height - ws.root1) / (90.0 - ws.root1)) +
             p3 * log((height - ws.root2) / (90.0 - ws.root2)) +
             p4 * log(
        (height^2 - 2.0 * ws.x_root * height + roots_2) /
        (8100.0 - 180.0 * ws.x_root + roots_2),
    )

    f2 = (height - 90.0) * (
        _JR_M_CON[7] + p5 / ((height + R_polar_km) * (90.0 + R_polar_km))
    ) + p6 * atan(
        ws.y_root * (height - 90.0) /
        (ws.y_root^2 + (height - ws.x_root) * (90.0 - ws.x_root)),
    ) / ws.y_root

    factor_k = -_JR_G_ZERO / (_JR_GAS_CON * (ws.tx - _JR_TZERO))
    return _JR_RHO_ZERO * _JR_TZERO * m_poly * exp(factor_k * (log_f1 + f2)) /
           (_JR_MZERO * temperature)
end


function rho_125(
    height::Real,
    temperature::Real,
    ws::JacchiaRobertsWorkspace,
    R_polar_km::Real,
)
    rho_prime = _horner(_JR_ZETA_CON, ws.t_infinity)
    t_100 = ws.tx + _JR_OMEGA * (ws.tx - _JR_TZERO)

    roots_2 = ws.x_root^2 + ws.y_root^2
    x_star = -2.0 * ws.root1 * ws.root2 * R_polar_km * (
        R_polar_km^2 + 2.0 * R_polar_km * ws.x_root + roots_2
    )
    v = (R_polar_km + ws.root1) * (R_polar_km + ws.root2) * (
        R_polar_km^2 + 2.0 * R_polar_km * ws.x_root + roots_2
    )
    u1 = (ws.root1 - ws.root2) * (ws.root1 + R_polar_km)^2 *
         (ws.root1^2 - 2.0 * ws.root1 * ws.x_root + roots_2)
    u2 = (ws.root1 - ws.root2) * (ws.root2 + R_polar_km)^2 *
         (ws.root2^2 - 2.0 * ws.root2 * ws.x_root + roots_2)
    w1 = ws.root1 * ws.root2 * R_polar_km * (R_polar_km + ws.root1) *
         (R_polar_km + roots_2 / ws.root1)
    w2 = ws.root1 * ws.root2 * R_polar_km * (R_polar_km + ws.root2) *
         (R_polar_km + roots_2 / ws.root2)

    q2 = 1.0 / u1
    q3 = -1.0 / u2
    q5 = 1.0 / v
    q4 = (1.0 + w1 * q2 + w2 * q3 + ws.root1 * ws.root2 * (R_polar_km^2 - roots_2) * q5) / x_star
    q1 = -2 * q4 - q3 - q2
    q6 = -q5 - 2.0 * (ws.x_root + R_polar_km) * q4 - (ws.root2 + R_polar_km) * q3 -
         (ws.root1 + R_polar_km) * q2

    log_f3 = q1 * log((height + R_polar_km) / (100.0 + R_polar_km)) +
             q2 * log((height - ws.root1) / (100.0 - ws.root1)) +
             q3 * log((height - ws.root2) / (100.0 - ws.root2)) +
             q4 * log(
        (height^2 - 2.0 * ws.x_root * height + roots_2) /
        (1.0e4 - 200.0 * ws.x_root + roots_2),
    )

    f4 = (height - 100.0) * q5 / ((height + R_polar_km) * (100.0 + R_polar_km)) +
         q6 * atan(
        ws.y_root * (height - 100.0) /
        (ws.y_root^2 + (height - ws.x_root) * (100.0 - ws.x_root)),
    ) / ws.y_root

    factor_k = -1.500625e6 * _JR_G_ZERO * R_polar_km^2 / (_JR_GAS_CON * _JR_CON_C[5] * (ws.tx - _JR_TZERO))

    rho_sum = 0.0
    for i in 1:5
        rhoi = _JR_MOL_MASS[i] * _JR_NUM_DENS[i] *
               exp(_JR_MOL_MASS[i] * factor_k * (f4 + log_f3))
        if i == 3
            rhoi *= (t_100 / temperature)^(-0.38)
        end
        rho_sum += rhoi
    end
    return rho_sum * rho_prime * t_100 / temperature
end


function rho_high(
    height::Real,
    temperature::Real,
    t_500::Real,
    sun_dec::Real,
    geo_lat::Real,
    ws::JacchiaRobertsWorkspace,
    R_polar_km::Real,
)
    rho_out = 0.0
    polar125 = R_polar_km + 125.0
    for i in 1:6
        if i <= 5
            log_di = _horner(_JR_CON_DEN[i], ws.t_infinity)
            di = 10.0^log_di / _JR_AVOGADRO
        end
        gamma = 35.0 * _JR_MOL_MASS[i] * _JR_G_ZERO * R_polar_km^2 * (ws.t_infinity - ws.tx) /
                (_JR_GAS_CON * ws.sum_L * ws.t_infinity * (ws.tx - _JR_TZERO) * polar125)
        exp1 = 1.0 + gamma
        f = 1.0
        if i == 3
            exp1 -= 0.38
            if abs(sun_dec) > 1.0e-15
                f = 4.9914 * abs(sun_dec) * (
                    sin(0.25 * _JR_PI - 0.5 * geo_lat * sun_dec / abs(sun_dec))^3 - 0.35355
                ) / _JR_PI
            else
                f = 1.0
            end
            f = 10.0^f
        end
        if height > 500.0 && i == 6
            r = _JR_MOL_MASS[6] * 10.0^(73.13 - (39.4 - 5.5 * log10(t_500)) * log10(t_500)) *
                (t_500 / temperature)^exp1 *
                ((ws.t_infinity - temperature) / (ws.t_infinity - t_500))^gamma / _JR_AVOGADRO
            rho_out += r
        elseif i <= 5
            r = f * _JR_MOL_MASS[i] * di * (ws.tx / temperature)^exp1 *
                ((ws.t_infinity - temperature) / (ws.t_infinity - ws.tx))^gamma
            rho_out += r
        end
    end
    return rho_out
end


function rho_cor(
    height::Real, utc_mjd::Real, geo_lat::Real, geo::JacchiaRobertsGeomagneticExposphericParams,
)
    geo_cor = if height < 200.0
        0.012 * geo.tkp + 0.000012 * exp(geo.tkp)
    else
        0.0
    end
    day_58 = (utc_mjd - _JR_DAY58_EPOCH_MJD) / 365.2422
    tausa = day_58 + 0.09544 * ((0.5 * (1.0 + sin(2 * _JR_PI * day_58 + 6.035)))^1.65 - 0.5)
    alpha = sin(4.0 * _JR_PI * tausa + 4.259)
    g = 0.02835 + (0.3817 + 0.17829 * sin(2.0 * _JR_PI * tausa + 4.137)) * alpha
    semian_cor = (5.876e-7 * height^2.331 + 0.06328) * exp(-0.002868 * height) * g
    sin_lat = sin(geo_lat)
    eta_lat = sin(2.0 * _JR_PI * day_58 + 1.72) * sin_lat * abs(sin_lat)
    slat_cor = 0.014 * (height - 90.0) * eta_lat * exp(-0.0013 * (height - 90.0)^2)
    return 10.0^(geo_cor + semian_cor + slat_cor)
end


"""
    jacchia_roberts_density(height_km, r_km, sun_unit, utc_mjd, geo_lat_rad, geo, ws;
        R_polar_km=6356.766)

Compute Jacchia-Roberts atmospheric density in g/cm³.
"""
function jacchia_roberts_density(
    height_km::Real,
    r_km::AbstractVector{<:Real},
    sun_unit::AbstractVector{<:Real},
    utc_mjd::Real,
    geo_lat_rad::Real,
    geo::JacchiaRobertsGeomagneticExposphericParams,
    ws::JacchiaRobertsWorkspace;
    R_polar_km::Real = _JR_DEFAULT_R_POLAR_KM,
)
    sun_dec = atan(sun_unit[3], hypot(sun_unit[1], sun_unit[2]))
    R = float(R_polar_km)

    density = if height_km <= 90.0
        _JR_RHO_ZERO
    elseif height_km < 100.0
        temp = exotherm!(ws, r_km, sun_unit, geo, height_km, sun_dec, geo_lat_rad, R)
        rho_100(height_km, temp, ws, R)
    elseif height_km <= 125.0
        temp = exotherm!(ws, r_km, sun_unit, geo, height_km, sun_dec, geo_lat_rad, R)
        rho_125(height_km, temp, ws, R)
    elseif height_km <= 2500.0
        t_500 = exotherm!(ws, r_km, sun_unit, geo, 500.0, sun_dec, geo_lat_rad, R)
        temp = exotherm!(ws, r_km, sun_unit, geo, height_km, sun_dec, geo_lat_rad, R)
        rho_high(height_km, temp, t_500, sun_dec, geo_lat_rad, ws, R)
    else
        0.0
    end

    return density * rho_cor(height_km, utc_mjd, geo_lat_rad, geo)
end


"""
    JacchiaRobertsModel(height_km, r_km, sun_unit, utc_mjd, geo_lat_rad, geo, ws; kwargs...)

Atmospheric density from the Jacchia-Roberts model in kg/m³.
"""
function JacchiaRobertsModel(
    height_km::Real,
    r_km::AbstractVector{<:Real},
    sun_unit::AbstractVector{<:Real},
    utc_mjd::Real,
    geo_lat_rad::Real,
    geo::JacchiaRobertsGeomagneticExposphericParams,
    ws::JacchiaRobertsWorkspace;
    kwargs...,
)
    return 1e3 * jacchia_roberts_density(
        height_km, r_km, sun_unit, utc_mjd, geo_lat_rad, geo, ws; kwargs...
    )
end


"""
    jacchia_roberts_f_density(;
        frame_PCPF="IAU_EARTH",
        naif_frame="J2000",
        earth_id="399",
        abcorr="NONE",
        F107=150.0,
        F107a=150.0,
        Kp=3.0,
        R_polar_km=6356.766,
    )

Build an `f_density` callback using the Jacchia-Roberts model.

`r_km` passed to the callback must be in `frame_PCPF` (km), as supplied by the EOM.
`frame_PCPF` must match `params.frame_PCPF` used in propagation.
"""
function jacchia_roberts_f_density(;
    frame_PCPF::String = "IAU_EARTH",
    naif_frame::String = "J2000",
    earth_id::String = "399",
    abcorr::String = "NONE",
    F107::Real = 150.0,
    F107a::Real = 150.0,
    Kp::Real = 3.0,
    R_polar_km::Real = _JR_DEFAULT_R_POLAR_KM,
)
    geo = JacchiaRobertsGeomagneticExposphericParams(F107, F107a, Kp)
    ws = JacchiaRobertsWorkspace()
    Re = bodvrd(earth_id, "RADII", 3)[1]
    flat = (Re - bodvrd(earth_id, "RADII", 3)[3]) / Re
    R_polar_km = float(R_polar_km)

    function f_density(et, r_km_pcpf)
        lon, lat, alt = recgeo(r_km_pcpf, Re, flat)
        if alt <= 100.0
            error("Jacchia-Roberts atmosphere model is not available for altitudes below 100 km.")
        end
        r_sun, _ = spkpos("10", et, naif_frame, abcorr, earth_id)
        T = SPICE.pxform(naif_frame, frame_PCPF, et)
        sun = T * r_sun
        sun ./= norm(sun)
        utc_mjd = et_to_utc_mjd(et)
        rho_gcm3 = jacchia_roberts_density(
            alt, r_km_pcpf, sun, utc_mjd, lat, geo, ws; R_polar_km = R_polar_km,
        )
        return 1e3 * rho_gcm3
    end
    return f_density
end
