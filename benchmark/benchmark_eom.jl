"""Benchmark the N-body spherical harmonics equations of motion"""

using BenchmarkTools
using ForwardDiff
using LinearAlgebra
using OrdinaryDiffEq
using SPICE
using Test

include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
include(joinpath(@__DIR__, "../test/utils.jl"))

furnsh_kernels()


# benchmark_eom = function(;verbose::Bool = false)
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
nmax = 20

et0 = str2et("2020-01-01T00:00:00")
etf = et0 + 30 * 86400.0
parameters = HighFidelityEphemerisModel.SpiceParameters(
    et0, DU, GMs, naif_ids, naif_frame, abcorr;
    filepath_spherical_harmonics = filepath_spherical_harmonics,
    nmax = nmax,
    frame_PCPF = "MOON_PA")

# initial state (in canonical scale)
x0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]

# benchmark eom
dx = zeros(6)
@benchmark HighFidelityEphemerisModel.eom_NbodySH!(dx, x0, parameters, 0.0)