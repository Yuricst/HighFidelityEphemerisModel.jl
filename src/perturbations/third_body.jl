"""Perturbations due to third-body"""


"""
    third_body_accel(r_spacecraft, r_3body, mu_3body)

Compute third-body acceleration via Battin's formula (eqn (8.60) ~ (8.62) in [1])

[1] Battin, R. H. (1987). An introduction to the mathematics and methods of astrodynamics.
American Institute of Aeronautics and Astronautics, Inc.
"""
function third_body_accel(r_spacecraft::Vector, r_3body::Vector, mu_3body::Float64)
    q = dot(r_spacecraft, r_spacecraft - 2r_3body)/dot(r_3body, r_3body)
    F = q * (3 + 3q + q^2)/(1 + sqrt(1+q)^3)
    return -mu_3body/norm(r_spacecraft - r_3body)^3 * (r_spacecraft + F*r_3body)
end


"""
    third_body_accel!(dx, r_spacecraft, r_3body, mu_3body)

Compute third-body acceleration via Battin's formula (eqn (8.60) ~ (8.62) in [1])

[1] Battin, R. H. (1987). An introduction to the mathematics and methods of astrodynamics.
American Institute of Aeronautics and Astronautics, Inc.
"""
function third_body_accel!(dx, r_spacecraft::Vector, r_3body::Vector, mu_3body::Float64)
    q = dot(r_spacecraft, r_spacecraft - 2r_3body)/dot(r_3body, r_3body)
    F = q * (3 + 3q + q^2)/(1 + sqrt(1+q)^3)
    dx[4:6] += -mu_3body/norm(r_spacecraft - r_3body)^3 * (r_spacecraft + F*r_3body)
    return
end


"""
Traditional third-body acceleration formula, implemented for reference
"""
function third_body_accel_classical(r_spacecraft::Vector, r_3body::Vector, mu_3body::Float64)
    dr = r_spacecraft - r_3body
    return -mu_3body * (dr/norm(dr)^3 + r_3body/norm(r_3body)^3)
end