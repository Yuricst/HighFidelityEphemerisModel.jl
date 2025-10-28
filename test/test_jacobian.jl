"""Test sparse jacobian with autodiff"""


using LinearAlgebra
using SPICE
using Test

# if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
# end


include("utils.jl")
furnsh_kernels()


    # define parameters
    naif_ids = ["301", "399", "10"]
    GMs = [bodvrd(ID, "GM", 1)[1] for ID in naif_ids]
    naif_frame = "J2000"
    abcorr = "NONE"
    DU = 3000.0
    filepath_spherical_harmonics = joinpath(@__DIR__, "../data/luna/gggrx_1200l_sha_20x20.tab")
    nmax = 4

    et0 = str2et("2020-01-01T00:00:00")
    etf = et0 + 30 * 86400.0
    interpolate_ephem_span = [et0, etf]
    interpolation_time_step = 30.0
    parameters = HighFidelityEphemerisModel.HighFidelityEphemerisModelParameters(
        et0, DU, GMs, naif_ids, naif_frame, abcorr;
        interpolate_ephem_span=interpolate_ephem_span,
        filepath_spherical_harmonics = filepath_spherical_harmonics,
        nmax = nmax,
        frame_PCPF = "MOON_PA",
        interpolation_time_step = interpolation_time_step,
        include_srp = true,
        srp_Cr = 1.15,
        srp_Am = 0.002,
        srp_P0 = 4.56e-6,
    )
    HighFidelityEphemerisModel.set_sparse_jacobian_cache!(
        HighFidelityEphemerisModel.eom_NbodySH_Interp,
        parameters,
    )