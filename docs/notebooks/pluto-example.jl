### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ a70cef7d-2a2f-4155-bdf3-fec9df94c63f
begin
    using Pkg
	ENV["TORCH_USE_RTLD_GLOBAL"] = "1"

    Pkg.activate(joinpath(@__DIR__, ".."))
    using Revise
    using PyBayesOpt
    using CairoMakie
    using Optim
    using Distributions
end

# ╔═╡ baa3e08e-5d64-4c8f-9f6d-5fdb40e97bc5
begin
    using HypertextLiteral: @htl_str, @htl
    using UUIDs: uuid1
    using PlutoUI
end

# ╔═╡ 3f2a5921-48fb-4278-97cf-254e98671643
function f(x, p)
    return 10*p[1] + p[2] * x + p[3] * x^3 + p[4] * x^4 + p[5]*x^5
end

# ╔═╡ a139fee5-1fb2-4a0a-a356-d606ef65572f
p̂ = [0.5, 2, 0.1, -0.1, 0.05]

# ╔═╡ 806af05c-acfc-4e99-a21a-33172c4a1816
X = -5:0.1:5

# ╔═╡ 88e113c5-94db-4b31-92c4-cdee7a4ee3a9
nmeas = 20

# ╔═╡ 5d84e1a6-3424-4d86-aa15-d127137734f6
pertX=Normal(0,0.2)

# ╔═╡ 9b63b191-8851-45fd-874a-a7a316d73e18
Xmeas = [ x+rand(pertX) for x in -4.5:0.5:4.5] |>sort

# ╔═╡ 3c008c07-f6bc-4782-9032-1b0105ad3980
pertY = Normal(0, 3)

# ╔═╡ 64442eab-6337-4b34-a716-e48a10e4aa5e
Ymeas = [f(x, p̂) + rand(pertY) for x in Xmeas]

# ╔═╡ ddc68ad5-f6e1-40e0-891c-2cc9257d50d0
function lsq(p)
    res = 0.0
    for i in 1:length(Xmeas)
        res += (Ymeas[i] - f(Xmeas[i], p))^2
    end
    return res
end

# ╔═╡ 3aa6e1ce-6e51-4bfe-ba97-50ae8893b6c8
bounds = [fill(-1, 5) fill(1, 5)]

# ╔═╡ 745ca731-f47d-4570-a87e-f6c37f68ebfe
md"""
## BoTorch
"""

# ╔═╡ 46e85e36-f37b-4dbb-88f6-31c1b60eaea5
begin
    botopt = BoTorchOptimization(;
        bounds,
        seed = rand(10:1000),
        nbatch = 4,
        ninit = 10,
        nopt = 15,
        acq_nsamples = 1024,
        acqmethod = :qLogNEI
    )
    optimize!(botopt, p -> -lsq(p))
    nbo = botopt._evaluations_used
    pbotopt, vbotopt = bestpoint(botopt)
end

# ╔═╡ 541432aa-0e02-4522-983c-deb65df2ced8
md"""
# BayesianOptimization
"""

# ╔═╡ 8c9ea9b2-9a74-4ae1-be4f-f4196abe70f4
begin
	bayopt=BayesianOptimization(p->-lsq(p);
								bounds, ninit=40, nopt=60, verbose=1 )
	PyBayesOpt.maximize!(bayopt)
	nbay=PyBayesOpt.iterations(bayopt)
	pbayopt, vbayopt = bestpoint(bayopt)	
end

# ╔═╡ a27444c5-710b-4203-896a-7f4033bf5da9
md"""
## Nelder-Mead
"""

# ╔═╡ 631e0d25-998b-42a4-ac04-a0908af1a24d
@time nmres = optimize(
    lsq, zeros(length(p̂)), NelderMead(),
    Optim.Options(f_calls_limit = 100)
)

# ╔═╡ db099202-b424-49c5-9277-c5a43ab324be
pnm = Optim.minimizer(nmres)

# ╔═╡ f4b3b38e-cefd-488c-b738-2cf4b93f637b
nnm = Optim.f_calls(nmres)

# ╔═╡ c373f697-fc1e-43f2-a590-f84489398128
let
    fig = Figure(size = (600, 300))
    ax = Axis(fig[1, 1])
    lines!(ax, X, [f(x, p̂) for x in X], label = "exact", 
		   color = :black)
    scatter!(ax, Xmeas, Ymeas, 
			 color = :red, label = "meas")
    lines!(ax, X, [f(x, pnm) for x in X], 
		   color = :green, label = "Nelder-Mead($(nnm))")
    lines!(ax, X, [f(x, pbotopt) for x in X], 		   color = :blue, label = "BoTorch($(nbo))")
    lines!(ax, X, [f(x, pbayopt) for x in X], 		   color = :orange, label = "bayes_opt($(nbay))")
    axislegend(position = :cb)
    fig
end

# ╔═╡ 8af12f1c-d35b-4cc9-8185-1bb5adbb69e8
html"""<hr>"""

# ╔═╡ 784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
html"""<style>.dont-panic{ display: none }</style>"""

# ╔═╡ afe4745f-f9f1-4e23-8735-cbec6fb79c41
begin
    function floataside(text::Markdown.MD; top = 1)
        uuid = uuid1()
        return @htl(
            """
            		<style>


            		@media (min-width: calc(700px + 30px + 300px)) {
            			aside.plutoui-aside-wrapper-$(uuid) {

            	color: var(--pluto-output-color);
            	position:fixed;
            	right: 1rem;
            	top: $(top)px;
            	width: 400px;
            	padding: 10px;
            	border: 3px solid rgba(0, 0, 0, 0.15);
            	border-radius: 10px;
            	box-shadow: 0 0 11px 0px #00000010;
            	/* That is, viewport minus top minus Live Docs */
            	max-height: calc(100vh - 5rem - 56px);
            	overflow: auto;
            	z-index: 40;
            	background-color: var(--main-bg-color);
            	transition: transform 300ms cubic-bezier(0.18, 0.89, 0.45, 1.12);

            			}
            			aside.plutoui-aside-wrapper > div {
            #				width: 300px;
            			}
            		}
            		</style>

            		<aside class="plutoui-aside-wrapper-$(uuid)">
            		<div>
            		$(text)
            		</div>
            		</aside>

            		"""
        )
    end
    floataside(stuff; kwargs...) = floataside(md"""$(stuff)"""; kwargs...)
end;


# ╔═╡ b8fd36a7-d8d1-45f7-b66e-df9132168bfc
# https://discourse.julialang.org/t/adding-a-restart-process-button-in-pluto/76812/5
restart_button() = html"""
<script>
	const button = document.createElement("button")

	button.addEventListener("click", () => {
		editor_state_set(old_state => ({
			notebook: {
				...old_state.notebook,
				process_status: "no_process",
			},
		})).then(() => {
			window.requestAnimationFrame(() => {
				document.querySelector("#process_status a").click()
			})
		})
	})
	button.innerText = "Restart notebook"

	return button
</script>
""";

# ╔═╡ Cell order:
# ╠═a70cef7d-2a2f-4155-bdf3-fec9df94c63f
# ╠═3f2a5921-48fb-4278-97cf-254e98671643
# ╠═a139fee5-1fb2-4a0a-a356-d606ef65572f
# ╠═806af05c-acfc-4e99-a21a-33172c4a1816
# ╠═88e113c5-94db-4b31-92c4-cdee7a4ee3a9
# ╠═5d84e1a6-3424-4d86-aa15-d127137734f6
# ╠═9b63b191-8851-45fd-874a-a7a316d73e18
# ╠═3c008c07-f6bc-4782-9032-1b0105ad3980
# ╠═64442eab-6337-4b34-a716-e48a10e4aa5e
# ╠═ddc68ad5-f6e1-40e0-891c-2cc9257d50d0
# ╠═3aa6e1ce-6e51-4bfe-ba97-50ae8893b6c8
# ╟─c373f697-fc1e-43f2-a590-f84489398128
# ╠═745ca731-f47d-4570-a87e-f6c37f68ebfe
# ╠═46e85e36-f37b-4dbb-88f6-31c1b60eaea5
# ╟─541432aa-0e02-4522-983c-deb65df2ced8
# ╠═8c9ea9b2-9a74-4ae1-be4f-f4196abe70f4
# ╟─a27444c5-710b-4203-896a-7f4033bf5da9
# ╠═631e0d25-998b-42a4-ac04-a0908af1a24d
# ╠═db099202-b424-49c5-9277-c5a43ab324be
# ╠═f4b3b38e-cefd-488c-b738-2cf4b93f637b
# ╟─8af12f1c-d35b-4cc9-8185-1bb5adbb69e8
# ╟─baa3e08e-5d64-4c8f-9f6d-5fdb40e97bc5
# ╟─784b4c3e-bb2a-4940-a83a-ed5e5898dfd4
# ╟─afe4745f-f9f1-4e23-8735-cbec6fb79c41
# ╟─b8fd36a7-d8d1-45f7-b66e-df9132168bfc
