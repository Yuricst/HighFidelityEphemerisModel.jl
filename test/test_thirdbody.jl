"""Test for third-body perturbation"""

using Random
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


function test_third_body_accel()
    r_spacecraft = [1.0, 0.1, -0.34]
    r_3body = [1.0, 4.0, -0.4]
    mu_3body = 0.2

    for N_try in 1:10
        Random.seed!(N_try)
        
        r_spacecraft = randn(3)
        r_3body = randn(3)
        mu_3body = rand()

        accel = HighFidelityEphemerisModel.third_body_accel(r_spacecraft, r_3body, mu_3body)
        accel_classical = HighFidelityEphemerisModel.third_body_accel_classical(r_spacecraft, r_3body, mu_3body)
        # println()
        # @show accel
        # @show accel_classical
        # @show accel2
        @test accel ≈ accel_classical atol=1e-14
    end
end

test_third_body_accel()