"""Test for ForwardDiff-based Hessian evaluation"""

using LinearAlgebra
using SPICE
using Test

if !@isdefined(HighFidelityEphemerisModel)
    include(joinpath(@__DIR__, "../src/HighFidelityEphemerisModel.jl"))
end


test_hessian_fd = function(;verbose::Bool = false)
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

    # initial state (in canonical scale)
    x0 = [1.0, 0.0, 0.3, 0.5, 1.0, 0.0]
    x0_stm = [x0; reshape(I(6),36)]

    # list of eom's to check for hessian evaluation
    eom_list = [
        HighFidelityEphemerisModel.eom_Nbody_SPICE,
        HighFidelityEphemerisModel.eom_Nbody_Interp,
        HighFidelityEphemerisModel.eom_NbodySH_SPICE,
        HighFidelityEphemerisModel.eom_NbodySH_Interp,
    ]

    for eom in eom_list
        if verbose
            println("\nTesting Hessian evaluation for $eom")
        end
        # evaluate Hessian via numerical difference of Jacobians
        jac_numerical = HighFidelityEphemerisModel.eom_jacobian_fd(
            eom, x0, 0.0, parameters, 0.0
        )
        hess_numerical = zeros(6,6,6)
        h = 1e-8
        for i = 1:6
            x0_copy = copy(x0)
            x0_copy[i] += h
            jac_ptrb = HighFidelityEphemerisModel.eom_jacobian_fd(
                eom, x0_copy, 0.0, parameters, 0.0
            )
            hess_numerical[:,:,i] = (jac_ptrb - jac_numerical) / h
        end

        # evaluate Hessian via ForwardDiff
        hess_fd = HighFidelityEphemerisModel.eom_hessian_fd(
            eom, x0, 0.0, parameters, 0.0
        )

        # if verbose
        #     println("ForwardDiff Hessian:")
        #     for i = 1:6
        #         println("i = $i")
        #         print_matrix(hess_fd[i,:,:])
        #     end
        #     println()
        #     println("Numerical Hessian:")
        #     for i = 1:6
        #         println("i = $i")
        #         print_matrix(hess_numerical[i,:,:])
        #     end
        #     println()
        # end

        for i = 1:6
            if verbose
                @show maximum(abs.(hess_fd[i,:,:] - hess_numerical[i,:,:]))
            end
            @test hess_fd[i,:,:] ≈ hess_numerical[i,:,:] atol = 1e-7
        end
    end
end

test_hessian_fd(verbose = false)