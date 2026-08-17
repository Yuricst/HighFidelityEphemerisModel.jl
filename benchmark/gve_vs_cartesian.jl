"""Benchmark Cartesian and Gauss variational equations of motion."""

using AstrodynamicsCore
using BenchmarkTools
using HighFidelityEphemerisModel
using LinearAlgebra
using OrdinaryDiffEq
using Printf
using SPICE

include(joinpath(@__DIR__, "../test/utils.jl"))
furnsh_kernels()


function trajectory_errors(sol_rv, sol_elements, elements2rv, mu, ts)
    position_errors = Float64[]
    velocity_errors = Float64[]
    for t in ts
        rv_elements = elements2rv(sol_elements(t), mu)
        push!(position_errors, norm(rv_elements[1:3] - sol_rv(t)[1:3]))
        push!(velocity_errors, norm(rv_elements[4:6] - sol_rv(t)[4:6]))
    end
    return position_errors, velocity_errors
end


function propagation_benchmark(eom, gve, rv0, elements0, elements2rv, params, tspan)
    ts = range(tspan[1], tspan[2], length = 101)
    prob_rv = ODEProblem(eom, rv0, tspan, params)
    prob_elements = ODEProblem(gve, elements0, tspan, params)

    solve(prob_rv, Vern9(), reltol = 1e-12, abstol = 1e-12, saveat = ts)
    solve(prob_elements, Vern9(), reltol = 1e-12, abstol = 1e-12, saveat = ts)
    timing_rv = @timed solve(prob_rv, Vern9(), reltol = 1e-12, abstol = 1e-12, saveat = ts)
    timing_elements = @timed solve(prob_elements, Vern9(), reltol = 1e-12, abstol = 1e-12, saveat = ts)
    position_errors, velocity_errors = trajectory_errors(
        timing_rv.value, timing_elements.value, elements2rv, params.mus[1], ts)

    return (
        cartesian_time = timing_rv.time,
        cartesian_bytes = timing_rv.bytes,
        element_time = timing_elements.time,
        element_bytes = timing_elements.bytes,
        final_position_error = position_errors[end],
        final_velocity_error = velocity_errors[end],
        max_position_error = maximum(position_errors),
        max_velocity_error = maximum(velocity_errors),
    )
end


naif_ids = ["301", "399", "10"]
GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
et0 = str2et("2020-01-01T00:00:00")
DU = 3000.0
filepath_SH = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
params = SpiceParameters(et0, DU, GMs, naif_ids, "J2000", "NONE")
params_SH = SpiceParameters(et0, DU, GMs, naif_ids, "J2000", "NONE";
    filepath_spherical_harmonics = filepath_SH, nmax = 8, frame_PCPF = "MOON_PA")

rv0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]
mee0 = AstrodynamicsCore.rv2mee(rv0, params.mus[1])
kep0 = AstrodynamicsCore.rv2kep(rv0, params.mus[1])
eq0 = AstrodynamicsCore.rv2eq(rv0, params.mus[1])
tspan = (0.0, 2.0)

dx_rv = zeros(6)
dx_mee = zeros(6)
dx_kep = zeros(6)
dx_eq = zeros(6)
eom_Nbody!(dx_rv, rv0, params, 0.0)
gve_mee_Nbody!(dx_mee, mee0, params, 0.0)
gve_kep_Nbody!(dx_kep, kep0, params, 0.0)
gve_eq_Nbody!(dx_eq, eq0, params, 0.0)
eom_NbodySH!(dx_rv, rv0, params_SH, 0.0)
gve_mee_NbodySH!(dx_mee, mee0, params_SH, 0.0)
gve_kep_NbodySH!(dx_kep, kep0, params_SH, 0.0)
gve_eq_NbodySH!(dx_eq, eq0, params_SH, 0.0)

a_inr = [0.3, -0.2, 0.7]
a_rtn = zeros(3)
rview = view(rv0, 1:3)
vview = view(rv0, 4:6)
project_inr_to_rtn!(a_rtn, a_inr, rview, vview)
HighFidelityEphemerisModel._project_inr_to_rtn_mee!(a_rtn, a_inr, mee0)

eval_cart = @benchmark eom_Nbody!($dx_rv, $rv0, $params, 0.0) samples = 100 seconds = 2
eval_mee = @benchmark gve_mee_Nbody!($dx_mee, $mee0, $params, 0.0) samples = 100 seconds = 2
eval_kep = @benchmark gve_kep_Nbody!($dx_kep, $kep0, $params, 0.0) samples = 100 seconds = 2
eval_eq = @benchmark gve_eq_Nbody!($dx_eq, $eq0, $params, 0.0) samples = 100 seconds = 2
eval_cart_SH = @benchmark eom_NbodySH!($dx_rv, $rv0, $params_SH, 0.0) samples = 100 seconds = 2
eval_mee_SH = @benchmark gve_mee_NbodySH!($dx_mee, $mee0, $params_SH, 0.0) samples = 100 seconds = 2
eval_kep_SH = @benchmark gve_kep_NbodySH!($dx_kep, $kep0, $params_SH, 0.0) samples = 100 seconds = 2
eval_eq_SH = @benchmark gve_eq_NbodySH!($dx_eq, $eq0, $params_SH, 0.0) samples = 100 seconds = 2
projection_cartesian = @benchmark project_inr_to_rtn!($a_rtn, $a_inr, $rview, $vview) samples = 1000 seconds = 2
projection_mee = @benchmark HighFidelityEphemerisModel._project_inr_to_rtn_mee!(
    $a_rtn, $a_inr, $mee0) samples = 1000 seconds = 2

initial_conditions = [
    ("general", rv0),
    ("near-circular planar",
        AstrodynamicsCore.kep2rv([1.5, 0.01, 1e-4, 0.2, 0.4, 0.6], params.mus[1])),
    ("eccentric inclined",
        AstrodynamicsCore.kep2rv([2.0, 0.3, 0.6, 1.0, 0.5, 2.0], params.mus[1])),
]
propagation_results = []
for (model, eom, parameters, mee_gve, kep_gve, eq_gve) in (
        ("N-body", eom_Nbody!, params, gve_mee_Nbody!, gve_kep_Nbody!, gve_eq_Nbody!),
        ("N-body+SH", eom_NbodySH!, params_SH, gve_mee_NbodySH!, gve_kep_NbodySH!, gve_eq_NbodySH!),
    ), (formulation, gve, rv2elements, elements2rv) in (
        ("MEE", mee_gve, AstrodynamicsCore.rv2mee, AstrodynamicsCore.mee2rv),
        ("Keplerian", kep_gve, AstrodynamicsCore.rv2kep, AstrodynamicsCore.kep2rv),
        ("ordinary equinoctial", eq_gve, AstrodynamicsCore.rv2eq, AstrodynamicsCore.eq2rv),
    ), (case_name, initial_rv) in initial_conditions
    initial_elements = rv2elements(initial_rv, parameters.mus[1])
    result = propagation_benchmark(
        eom, gve, initial_rv, initial_elements, elements2rv, parameters, tspan)
    all(isfinite, result) || error("GVE benchmark produced a nonfinite result")
    result.max_position_error < 1e-7 || error("GVE position error exceeded 1e-7 DU")
    result.max_velocity_error < 1e-7 || error("GVE velocity error exceeded 1e-7 VU")
    push!(propagation_results, (; model, formulation, case_name, result...))
end

output = get(ENV, "GVE_BENCHMARK_OUTPUT",
    joinpath(@__DIR__, "reports", "gve_vs_cartesian.md"))
mkpath(dirname(output))
open(output, "w") do io
    println(io, "# GVE versus Cartesian benchmark")
    println(io)
    println(io, "Julia: `$(VERSION)`")
    println(io)
    println(io, "> These measurements describe this trajectory set, backend, solver, " *
        "tolerances, Julia version, and hardware; they are not universal performance guarantees.")
    println(io, "> Cartesian and element integrations use adaptive error control in different " *
        "coordinates, so step sequences and timing can differ with coordinate conditioning.")
    println(io)
    println(io, "## EOM evaluation")
    println(io)
    println(io, "| Model | Minimum time (ns) | Allocations | Memory (bytes) |")
    println(io, "|---|---:|---:|---:|")
    for (name, trial) in (("Cartesian N-body", eval_cart), ("MEE N-body", eval_mee),
            ("Keplerian N-body", eval_kep), ("ordinary equinoctial N-body", eval_eq),
            ("Cartesian N-body+SH", eval_cart_SH), ("MEE N-body+SH", eval_mee_SH),
            ("Keplerian N-body+SH", eval_kep_SH),
            ("ordinary equinoctial N-body+SH", eval_eq_SH))
        estimate = minimum(trial)
        @printf(io, "| %s | %.0f | %d | %d |\n",
            name, estimate.time, estimate.allocs, estimate.memory)
    end
    println(io)
    println(io, "## RTN projection")
    println(io)
    println(io, "| Projection | Minimum time (ns) | Allocations | Memory (bytes) |")
    println(io, "|---|---:|---:|---:|")
    for (name, trial) in (("Cartesian-derived", projection_cartesian),
            ("Direct MEE", projection_mee))
        estimate = minimum(trial)
        @printf(io, "| %s | %.0f | %d | %d |\n",
            name, estimate.time, estimate.allocs, estimate.memory)
    end
    println(io)
    println(io, "## Integrated propagation")
    println(io)
    println(io, "| Model | Formulation | Initial condition | Cartesian time (s) | Element time (s) | Cartesian bytes | Element bytes | Final position error | Final velocity error | Max position error | Max velocity error |")
    println(io, "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in propagation_results
        @printf(io, "| %s | %s | %s | %.6f | %.6f | %d | %d | %.6e | %.6e | %.6e | %.6e |\n",
            result.model, result.formulation, result.case_name, result.cartesian_time,
            result.element_time, result.cartesian_bytes, result.element_bytes,
            result.final_position_error, result.final_velocity_error,
            result.max_position_error, result.max_velocity_error)
    end
end

println(read(output, String))