"""Symbolic Jacobians for HighFidelityEphemerisModel"""


function symbolic_Nbody_jacobian(N::Int)
    # define symbolic variables
    Symbolics.@variables x y z #state[1:3]
    Symbolics.@variables mus[1:N]
    Symbolics.@variables Rs[1:3*(N-1)]

    # define accelerations
    rnorm = sqrt(x^2 + y^2 + z^2)
    ax = -mus[1]*x/rnorm^3
    ay = -mus[1]*y/rnorm^3
    az = -mus[1]*z/rnorm^3
    
    # third-body accelerations
    for i = 2:N
        R2 = Rs[1+3(i-2)]^2 + Rs[2+3(i-2)]^2 + Rs[3+3(i-2)]^2
        R3 = R2^(3/2)
        d3 = ((x - Rs[1+3(i-2)])^2 + (y - Rs[2+3(i-2)])^2 + (z - Rs[3+3(i-2)])^2)^(3/2)

        q = (x * (x - 2*Rs[1+3(i-2)]) +
             y * (y - 2*Rs[2+3(i-2)]) +
             z * (z - 2*Rs[3+3(i-2)])) / R2
        F = q * (3 + 3q + q^2)/(1 + sqrt(1+q)^3)

        ax += -mus[i] / d3 * (x + F*Rs[1+3(i-2)])
        ay += -mus[i] / d3 * (y + F*Rs[2+3(i-2)])
        az += -mus[i] / d3 * (z + F*Rs[3+3(i-2)])
    end

    Uxx = [
        Symbolics.derivative(ax, x), Symbolics.derivative(ax, y), Symbolics.derivative(ax, z),
        Symbolics.derivative(ay, x), Symbolics.derivative(ay, y), Symbolics.derivative(ay, z),
        Symbolics.derivative(az, x), Symbolics.derivative(az, y), Symbolics.derivative(az, z),
    ]

    arguments = [x, y, z, mus..., Rs...]
    f_Uxx = Symbolics.eval(Symbolics.build_function(Uxx, arguments...; expression = Val{false})[1])
    f_jacobian = function (rv, mus, Rs)
        jac = zeros(6,6)
        jac[1:3,4:6] = I(3)
        jac[4:6,1:3] = reshape(f_Uxx(rv[1:3]..., mus..., Rs...), (3,3))'
        return jac
    end
    return f_jacobian
end


function symbolic_NbodySRP_jacobian(N::Int)
    # define symbolic variables
    Symbolics.@variables x y z #state[1:3]
    Symbolics.@variables mus[1:N]
    Symbolics.@variables Rs[1:3*(N-1)]
    Symbolics.@variables r_sun[1:3]
    Symbolics.@variables k_srp_cannonball

    # define accelerations
    rnorm = sqrt(x^2 + y^2 + z^2)
    ax = -mus[1]*x/rnorm^3
    ay = -mus[1]*y/rnorm^3
    az = -mus[1]*z/rnorm^3
    
    # third-body accelerations
    for i = 2:N
        R2 = Rs[1+3(i-2)]^2 + Rs[2+3(i-2)]^2 + Rs[3+3(i-2)]^2
        R3 = R2^(3/2)
        d3 = ((x - Rs[1+3(i-2)])^2 + (y - Rs[2+3(i-2)])^2 + (z - Rs[3+3(i-2)])^2)^(3/2)

        q = (x * (x - 2*Rs[1+3(i-2)]) +
             y * (y - 2*Rs[2+3(i-2)]) +
             z * (z - 2*Rs[3+3(i-2)])) / R2
        F = q * (3 + 3q + q^2)/(1 + sqrt(1+q)^3)

        ax += -mus[i] / d3 * (x + F*Rs[1+3(i-2)])
        ay += -mus[i] / d3 * (y + F*Rs[2+3(i-2)])
        az += -mus[i] / d3 * (z + F*Rs[3+3(i-2)])
    end

    r_relative_sun = [x - r_sun[1], y - r_sun[2], z - r_sun[3]]
    ax += k_srp_cannonball * r_relative_sun[1] / norm(r_relative_sun)^3
    ay += k_srp_cannonball * r_relative_sun[2] / norm(r_relative_sun)^3
    az += k_srp_cannonball * r_relative_sun[3] / norm(r_relative_sun)^3

    Uxx = [
        Symbolics.derivative(ax, x), Symbolics.derivative(ax, y), Symbolics.derivative(ax, z),
        Symbolics.derivative(ay, x), Symbolics.derivative(ay, y), Symbolics.derivative(ay, z),
        Symbolics.derivative(az, x), Symbolics.derivative(az, y), Symbolics.derivative(az, z),
    ]

    arguments = [x, y, z, mus..., Rs..., k_srp_cannonball, r_sun...]
    f_Uxx = Symbolics.eval(Symbolics.build_function(Uxx, arguments...; expression = Val{false})[1])
    f_jacobian = function (rv, mus, Rs, k_srp_cannonball, r_sun)
        jac = zeros(6,6)
        jac[1:3,4:6] = I(3)
        jac[4:6,1:3] = reshape(f_Uxx(rv[1:3]..., mus..., Rs..., k_srp_cannonball, r_sun...), (3,3))'
        return jac
    end
    return f_jacobian
end

