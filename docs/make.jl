# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

using Documenter, PyBayesOpt
using Optim  # ensure Optim methods available during doc build

makedocs(
    modules = [PyBayesOpt],
    format = Documenter.HTML(; prettyurls = true),
    authors = "Jürgen Fuhrmann",
    sitename = "PyBayesOpt.jl",
    pages = Any["index.md", "internal.md"],
    repo = "https://github.com/j-fu/PyBayesOpt.jl",
    warnonly = [:missing_docs, :docs_block, :cross_references],
)

if !isinteractive()
    deploydocs(
        repo = "github.com/j-fu/PyBayesOpt.jl.git",
        push_preview = true
    )
end
