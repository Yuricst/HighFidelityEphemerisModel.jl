
function set_sparse_jacobian_cache!(
    eom::Function,
    params::HighFidelityEphemerisModelParameters,
    sd = SparseDiffTools.SymbolicsSparsityDetection(),
    adtype = SparseDiffTools.AutoSparse(SparseDiffTools.AutoFiniteDiff()),
)
    params.adtype = adtype
    params.jacobian_cache = SparseDiffTools.sparse_jacobian_cache(adtype, sd, x -> eom(x, params, 1.0), ones(6))
end
