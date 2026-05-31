"""Benchmark the Jacobian of the N-body problem"""

using BenchmarkTools
using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
include(joinpath(@__DIR__, "../test/utils.jl"))

furnsh_kernels()


benchmark_jacobian = function(;verbose::Bool = false)
    # define parameters
    GMs = [
        4.9028000661637961E+03,
        3.9860043543609598E+05,
        1.3271244004193938E+11,
    ]
    naif_ids = ["301", "399", "10"]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 3000.0
    filepath_spherical_harmonics = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    nmax = 4
    
    et0 = str2et("2020-01-01T00:00:00")
    etf = et0 + 30 * 86400.0
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        filepath_spherical_harmonics = filepath_spherical_harmonics,
        nmax = nmax,
        frame_PCPF = "MOON_PA")

    # initial state (in canonical scale)
    x0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]
    x0_stm = [x0; reshape(I(6),36)]

    # evaluate Jacobian
    jac_analytical = HighFidelityEphemerisModel.dfdx_Nbody_SPICE(x0, 0.0, parameters, 0.0)

    f_eval = zeros(6)
    HighFidelityEphemerisModel.eom_Nbody_SPICE!(f_eval, x0, parameters, 0.0)
    jac_numerical = zeros(6,6)
    h = 1e-8
    for i = 1:6
        x0_copy = copy(x0)
        x0_copy[i] += h
        _f_eval = zeros(6)
        HighFidelityEphemerisModel.eom_Nbody_SPICE!(_f_eval, x0_copy, parameters, 0.0)
        jac_numerical[:,i] = (_f_eval - f_eval) / h
    end
    jac_numerical_fd = HighFidelityEphemerisModel.eom_jacobian_fd(
        HighFidelityEphemerisModel.eom_Nbody_SPICE, x0, 0.0, parameters, 0.0
    )

    # jacobian via ForwardDiff
    println("Analytical Jacobian:")
    print_matrix(jac_analytical)
    println()
    println("Numerical Jacobian:")
    print_matrix(jac_numerical)
    println()
    println("ForwardDiff Jacobian:")
    print_matrix(jac_numerical_fd)
    println()
    @test jac_analytical ≈ jac_numerical atol=1e-6
    @test jac_analytical ≈ jac_numerical_fd atol=1e-12

    # benchmarks
    open(joinpath(@__DIR__, "reports/benchmark_Nbody_jacobian.md"), "w+") do f
        write(f, "# Benchmarking N-body Jacobian\n\n")

        println("Benchmarking analytical Jacobian:")
        io = IOBuffer()
        show(io, "text/plain", @benchmark HighFidelityEphemerisModel.dfdx_Nbody_SPICE($x0, 0.0, $parameters, 0.0))

        write(f, "\n## N-body Jacobian with analytical method\n\n")
        write(f, "```julia\n@benchmark HighFidelityEphemerisModel.dfdx_Nbody_SPICE($x0, 0.0, $parameters, 0.0)\n```\n")
        write(f, "```\n"*String(take!(io))*"\n```\n\n")
    
        println("Benchmarking ForwardDiff Jacobian:")
        io = IOBuffer()
        show(io, "text/plain", @benchmark HighFidelityEphemerisModel.eom_jacobian_fd(
            HighFidelityEphemerisModel.eom_Nbody_SPICE, $x0, 0.0, $parameters, 0.0
        ))

        write(f, "\n## N-body Jacobian with ForwardDiff\n\n")
        write(f, "```julia\n@benchmark HighFidelityEphemerisModel.eom_jacobian_fd(HighFidelityEphemerisModel.eom_Nbody_SPICE, $x0, 0.0, $parameters, 0.0)\n```\n")
        write(f, "```\n"*String(take!(io))*"\n```\n\n")
    end
end


benchmark_jacobian()