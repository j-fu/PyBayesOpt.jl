# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

using Documenter, BoTorchOpt

makedocs(
    modules = [BoTorchOpt],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Jürgen Fuhrmann",
    sitename = "BoTorchOpt.jl",
    pages = Any["index.md"],
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

# Some setup is needed for documentation deployment, see “Hosting Documentation” and
# deploydocs() in the Documenter manual for more information.
deploydocs(
    repo = "github.com/j-fu/BoTorchOpt.jl.git",
    push_preview = true
)
