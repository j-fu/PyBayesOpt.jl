# see documentation at https://juliadocs.github.io/Documenter.jl/stable/

using Documenter, PyBayesOpt
using Optim  # ensure Optim methods available during doc build

is_ci = get(ENV, "CI", "false") == "true"

if is_ci
    makedocs(
        modules = [PyBayesOpt],
        format = Documenter.HTML(; prettyurls = true),
        authors = "Jürgen Fuhrmann",
        sitename = "PyBayesOpt.jl",
        pages = Any["index.md", "internal.md"],
        repo = "https://github.com/j-fu/PyBayesOpt.jl",
        warnonly = [:missing_docs, :docs_block, :cross_references],
    )
else
    makedocs(
        modules = [PyBayesOpt],
        format = Documenter.HTML(; prettyurls = false, edit_link = nothing),
        authors = "Jürgen Fuhrmann",
        sitename = "PyBayesOpt.jl",
        pages = Any["index.md", "internal.md"],
        repo = "local", # dummy
        remotes = Dict(),
        warnonly = [:missing_docs, :docs_block, :cross_references],
    )
end

if is_ci
    deploydocs(
        repo = "github.com/j-fu/PyBayesOpt.jl.git",
        push_preview = true
    )
end
