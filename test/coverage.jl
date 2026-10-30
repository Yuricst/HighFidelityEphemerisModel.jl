"""Check line coverage for `src` after running tests with `--code-coverage=user`."""

using Coverage
using Test

const SRC_DIR = normpath(joinpath(@__DIR__, "..", "src"))
const MIN_COVERAGE = parse(Float64, get(ENV, "HFEM_MIN_COVERAGE", "0.80"))

covered, total = get_summary(process_folder(SRC_DIR))
fraction = covered / total

@info "HighFidelityEphemerisModel src coverage" covered total percent=round(100 * fraction; digits=2)

for (root, _, files) in walkdir(SRC_DIR)
    for file in files
        endswith(file, ".jl") || continue
        path = joinpath(root, file)
        file_covered, file_total = get_summary(process_file(path))
        file_total == 0 && continue
        @debug relpath(path, SRC_DIR) covered=file_covered total=file_total percent=round(100 * file_covered / file_total; digits=1)
    end
end

@test fraction >= MIN_COVERAGE