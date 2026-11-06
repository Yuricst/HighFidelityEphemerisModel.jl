"""Utils for test"""

using Printf

function print_matrix(A)
    for i in 1:size(A,1)
        for j in 1:size(A,2)
            @printf("% 1.6e  ", A[i,j])
        end
        println()
    end
end


function furnsh_kernels()
    if !haskey(ENV, "SPICE")
        spice_dir = joinpath(@__DIR__, "../spice/test")
        furnsh(joinpath(spice_dir, "naif0012.tls"))
        furnsh(joinpath(spice_dir, "de440.bsp"))
        furnsh(joinpath(spice_dir, "gm_de440.tpc"))
        furnsh(joinpath(spice_dir, "earth_latest_high_prec.bpc"))
        furnsh(joinpath(spice_dir, "pck00011.tpc"))
        furnsh(joinpath(spice_dir, "moon_pa_de440_200625.bpc"))
        furnsh(joinpath(spice_dir, "moon_de440_250416.tf"))
        furnsh(joinpath(spice_dir, "receding_horiz_3189_1burnApo_DiffCorr_15yr.bsp"))
    else
        spice_dir = ENV["SPICE"]
        furnsh(joinpath(spice_dir, "lsk", "naif0012.tls"))
        furnsh(joinpath(spice_dir, "spk", "de440.bsp"))
        furnsh(joinpath(spice_dir, "pck", "gm_de440.tpc"))
        furnsh(joinpath(spice_dir, "pck", "earth_latest_high_prec.bpc"))
        furnsh(joinpath(spice_dir, "pck", "pck00011.tpc"))
        furnsh(joinpath(spice_dir, "pck", "moon_pa_de440_200625.bpc"))
        furnsh(joinpath(spice_dir, "fk", "moon_de440_250416.tf"))
        furnsh(joinpath(spice_dir, "misc", "dsg_naif", "receding_horiz_3189_1burnApo_DiffCorr_15yr.bsp"))
    end
end

"""
    hfem_test_kernel(kind, name)

Path of the test SPICE kernel `name`, which lives in `ENV["SPICE"]/kind/name` when the
`SPICE` environment variable is set and in `spice/test/name` otherwise (same layout as
`furnsh_kernels`). Used to build `EphemeridesParameters` from the same kernels.
"""
function hfem_test_kernel(kind::String, name::String)
    if haskey(ENV, "SPICE")
        return joinpath(ENV["SPICE"], kind, name)
    else
        return joinpath(@__DIR__, "../spice/test", name)
    end
end
