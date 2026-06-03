"""Run tests"""

using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

include("utils.jl")
include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))

# furnish spice kernels
furnsh_kernels()
verbose = false

@testset "Ephemeris interpolation" begin
    include("test_interpolate_ephem.jl")
    include("test_interpolate_transformation.jl")
end

@testset "N-body ODE             " begin
    include("test_thirdbody.jl")
    include("test_drag_harris_priester.jl")
    include("test_drag_jacchiaroberts.jl")
    include("test_Nbody_SPICE.jl")
    include("test_Nbody_Interp.jl")
    include("test_Nbody_ensemble.jl")
end

@testset "Spherical harmonics    " begin
    include("test_spherical_harmonics.jl")
    include("test_NbodySH_SPICE.jl")
    include("test_NbodySH_Interp.jl")
    include("test_NbodySH_ensemble.jl")
end

@testset "Callbacks              " begin
    include("test_callback.jl")
end

@testset "Hessian evaluation     " begin
    include("test_hessian_fd.jl")
end

@testset "SPK generation helpers " begin
    include("test_ode_sol_to_spk.jl")
end

if get(ENV, "JULIA_COVERAGE", "") == "true"
    @testset "Module coverage        " begin
        include("coverage.jl")
    end
end