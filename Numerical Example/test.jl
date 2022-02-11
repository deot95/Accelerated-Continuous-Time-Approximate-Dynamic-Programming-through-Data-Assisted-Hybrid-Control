#%%
################################################
############ Packages
################################################
include("../HybridSimulator.jl")
include("../PlantsInstances.jl")
include("../ActorCriticDynamics.jl")

using Plots
using FileIO
using Trapz
using HDF5
using Zygote
using LinearAlgebra
using DifferentialEquations
using Logging: global_logger
using TerminalLoggers: TerminalLogger
using Random
global_logger(TerminalLogger())

#%%
################################################
############ Subfolder creation
################################################
parentDir = @__DIR__
dataDir = "$parentDir/Data"
mkpath(dataDir)
figDir = "$parentDir/Figs"
mkpath(figDir)


#%%
################################################
############ Plant Dynamics
################################################

f = fKamalapurkar
g = gKamalapurkar

#%%
################################################
############ Cost functions
################################################
Π = 1
Q(x) = x'x
R(u) = u'Π*u


##Gotta Modify Analysis to include derivative of ζ
#=Π = 1
ū = 0.1
Q(x) = x'x
ζ(u) = ū*tanh(u/ū)
ζinv(u) = ū*atanh(u/ū)
rangeInt = 
R(u) = 2*quadgk(ζinv, 0, u)[1]
R(0.08)=#

#%%
################################################
############ Critic
################################################
function ϕc(x)
    x1, x2 = x
    return [x1^2; x1*x2; x2^2]
end
dϕc(x) = jacobian(ϕc, x)[1]
dϕc([1;1])
#%%
################################################
############ Actor
################################################

#%%
################################################
############ PESignal
################################################
pETurnOn = 0
function PESignal(t)
    # sbar=(sigmax/(sigmax'*sigmax+1)) should be PE
    probing_signal = exp(-0.001*t)*(sin(t)^2*cos(t) + 
    sin(2t)^2*cos(0.1t) +
    sin(-1.2t)^2*cos(0.5t) +
    sin(t)^5 + sin(1.12t)^2 +
    cos(2.4t)*sin(2.4t)^3*cos(0.01t)^6)
    return probing_signal*(t>pETurnOn)*(t<0)
end


################################################
############ Closed Loop
################################################
n = 2
m = 1
lc = 3
kc = 1#50
ku = 1#10
α₁ = 1
α₂ = 1
ρi = 1
ρd = 1
fModel = f
gModel = g

T₀ = 0.1
T = 5.5
α = 0


#Load collected data
fname = "$dataDir/SRFromOptimal.hdf5"
fid = h5open(fname, "r")
Λ = read(fid["Lambda"])
ψQRks = read(fid["psiQRks"])
close(fid) 

∇d(θ) = Λ*θ + ψQRks

#%%
#%%
################################################
############ Maps
################################################
maxT = 50
tspan = (0.0,maxT)
mag₀ = 10
magθc = 1
magθu = -1
x₀ = mag₀*[-1;-1]
θcStar = [1; 0.5; 0]
θc₀ = magθc*ones(lc, 1) #θcStar#

Random.seed!(1234)
θu₀ =  magθu*(0.5*ones(3) .+ 0.3(rand(3).-0.5))

lu = length(θu₀)
t₀ = 0
z₀ = [x₀; θc₀; vec(θu₀); t₀]

###Gradient
FCLInstance(u) = FClosedLoopGradientCriticAlt(u, n, lc, f, g, kc, ρi, ρd,
                                        dϕc, fModel, gModel, Q, R, ∇d, ku, α₁, α₂, Π, PESignal)
#FCLInstance(u) = FClosedLoopGradientCritic(u, n, m, lc, lu, f, g, kc, ρi, ρd,
                                        #dϕc, fModel, gModel, Q, R, ∇d, ku, ϕu, Π, (t->0)) 
function FCLGradient(du, u, p, t) 
    du[:] = FCLInstance(u)
end


###Hybrid
FCLHMInstance(z) = FClosedLoopHMCriticAlt(z, n, lc, f, g, kc, ρi, ρd,
                                        dϕc, fModel, gModel, Q, R, ∇d, ku, α₁, α₂, Π, PESignal)
#FCLHMInstance(z) = FClosedLoopHMCritic(z, n, m, lc, lu, f, g, kc, ρi, ρd,
#                                           dϕc, fModel, gModel, Q, R, ∇d, ku, ϕu, Π, (t->0))
function FCLHM(du, u, p, t) 
    du[:] = FCLHMInstance(u)
end

function GCLHM!(integrator)
    GClosedLoopHMCritic!(integrator, n, lc, α, T₀)
    @show integrator.u[n+2lc+1], integrator.t
end    
function condition(u,t,integrator) # Event when event_f(u,t) == 0
    return T-u[n+2lc+1]
end
cb = ContinuousCallback(condition, GCLHM!, rootfind=DiffEqBase.RightRootFind)

function resetInitialConditions!(integrator)
    integrator.u[1:n] = x₀
    @show integrator.u[1:n], integrator.t
end


#%%
################################################
############ Simulate Gradient Data
################################################
FCLInstance(z₀)

prob = ODEProblem(FCLGradient, z₀, tspan)
solGradient = solve(prob, Rodas4P(), progress = true,
                        progress_steps = 5)

l = @layout [a ; b]
p1 = plot(solGradient, vars=collect(n+1:n+lc), c=:gray, legend = false)
#plot!(solGradient, vars=collect(n+lc+1:n+lc+m*lu), c=:dodgerblue)
#hline!(θstar, ls=:dash, c=:darkred)

p2 = plot(solGradient, vars=(n+lc+1:n+lc+lu))
plot(p1, p2, layout = l)
solGradient(maxT)[n+lc+1:n+lc+lu]


p3 = plot(solGradient, vars=(1:n))

#%%
################################################
############ Simulate Momentum Data
################################################
tspan=(0.0, maxT)

pc₀ = θc₀
τ₀ = T₀
y₀ = [θc₀; pc₀; τ₀]
z₀ = [x₀; y₀; vec(θu₀); t₀]
FCLHMInstance(z₀)

prob = ODEProblem(FCLHM, z₀, tspan)
solHM= solve(prob, Rodas4P(), 
                callback=cb, progress = true,
                progress_steps = 5)
                
p1 = plot(solHM, vars=collect(n+1:n+lc), c=:darkgray, legend=false)

#p1 = plot(solHM, vars=collect(n+2lc+2: n+2lc+1+lu*m), c=:dodgerblue, legend=false)

p2 = plot(solHM, vars=(n+2lc+2:n+2lc+lu+1))
p3 = plot(solHM, vars=(1:n))
plot(p1, p2, layout=l)

plot(solHM, vars=(1:n))

### Comparison plot θc
plot(solGradient, vars=collect(n+1:n+lc), c=:darkgreen, legend = false)
plot!(solHM, vars=collect(n+1:n+lc), c=:darkblue, legend=false)
hline!(θcStar, c=:black, ls=:dash, legend=false)


### Comparison plot θu
tSpan = (0.0, maxT)
plot(solGradient, vars=collect(n+lc+1:n+lc+lu), tspan=tSpan, c=:darkgreen, legend = false)
plot!(solHM, vars=collect(n+2lc+2:n+2lc+1+lu), tspan=tSpan, c=:darkblue, legend=false)
hline!(θcStar, c=:black, ls=:dash, legend=false)

### System Dynamics Evolution
plot(solGradient, vars=collect(1:n), tspan=tSpan, c=:darkgreen, legend = false)
plot!(solHM, vars=collect(1:n), tspan=tSpan, c=:darkblue, legend=false)


# ts = range(0, stop=maxT, length=100).+0.001
# θcGradient = hcat(solGradient(ts).u...)[n+1:n+lc,:]
# θcHM = hcat(solHM(ts).u...)[n+1:n+lc,:]

# plot(ts, θcGradient', c=:darkgreen, legend = false, xaxis=:log10)
# plot!(ts, θcHM', c=:darkblue, legend=false, xaxis=:log10)

# plot(solHM, vars=(n+2lc+1))

timeD = collect(range(0, stop=maxT, length=100000))
arcGradient = hcat(solGradient(timeD).u...)
arcHM = hcat(solHM(timeD).u...)

################################################
############ Without Control
################################################
function fFreeKamalapurkar(dz, z, p, t)
    dz[:] = f(z)
end

prob = ODEProblem(fFreeKamalapurkar, x₀, tspan)
solFree = solve(prob, Rodas4P(), progress = true,
                        progress_steps = 5)
plot(solFree)
plot!(solGradient, vars=(1:n))
arcFree = hcat(solFree(timeD).u...)
################################################
############ Saving
################################################
fname = "$dataDir/resultsAltParam.hdf5"
fid = h5open(fname, "w")
fid["timeD"] = timeD
fid["arcGradient"] = arcGradient
fid["arcHM"] = arcHM
fid["arcFree"] = arcFree

close(fid)



