#%%
################################################
############ Packages
################################################
include("../HybridSimulator.jl")
include("../PlantsInstances.jl")

using Plots
using FileIO
using HDF5
using Zygote
using LinearAlgebra
using  StatsBase
using DifferentialEquations
import Random
using Logging: global_logger
using TerminalLoggers: TerminalLogger
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
############ Cost functions
################################################
Π = 1
Q(x) = x'x
R(u) = u'Π*u
ζ(x) = x
n = 2
m = 1
lc = 3
lu = 3

#%%
################################################
############ Maps and Probing Signal
################################################
function probingSignal(t)
    # sbar=(sigmax/(sigmax'*sigmax+1)) should be PE
    probing_signal = 37*exp(-0.001*t)*(sin(t)^2*cos(t) + 
                            sin(2t)^2*cos(0.1t) +
                            sin(-1.2t)^2*cos(0.5t) +
                            sin(t)^5 + sin(1.12t)^2 +
                            cos(2.4t)*sin(2.4t)^3*cos(0.01t)^6)
    return probing_signal
end

function fGenerateData(dz, z, p, t) 
    ####################
    ## State selection
    ####################
    z1, z2 = z
    u = -(cos(2z1) + 2)*z2 
    dz[:] = fKamalapurkar(z) + gKamlapurkar(z)*(u+probingSignal(t))
end

maxT = 500
tspan = (0, maxT)
z₀ = [1; -1]
prob = ODEProblem(fGenerateData, z₀, tspan)

solGenerateData = solve(prob, Tsit5(), progress = true,
                            progress_steps = 5)

plot(solGenerateData)

tHist = range(0, maxT, 1000)
xHist = hcat(solGenerateData(tHist).u...)

################################################
############ Auxiliary Functions
################################################
function ϕc(x)
    x1, x2 = x
    return [x1^2; x1*x2; x2^2]
end
dϕc(x) = jacobian(ϕc, x)[1]
dϕc([1;1])

u(x, t) =  -(cos(2x[1]) + 2)*x[2] + probingSignal(t)
ψ(x, u) =  dϕc(x)*(fKamalapurkar(x) + gKamlapurkar(x)*u)

#%%50######################
############ Data Sampling
################################################

nSamples = 36
Random.seed!(1234)
sampleIndices = sample(1:length(tHist), nSamples, replace=false)
xHistSample = xHist[:,sampleIndices]
tHistSample = tHist[sampleIndices]
uHistSample = [u(xk, tk) for (xk,tk) in zip(eachcol(xHistSample), tHistSample)]

Λ = zeros(lc, lc)
ψQRks = zeros(lc, 1)
for k in 1:nSamples
    xk,uk = xHistSample[:,k], uHistSample[k]
    ψk = ψ(xk,uk)
    nψk = 1 .+ ψk'ψk
    Ψk = ψk/nψk
    Λ +=  Ψk*Ψk'
    ψQRks += ψk*(Q(xk) + R(uk))/nψk^2
end
λ̲ = minimum(eigvals(Λ))

ψQRks = sum([ψk*(Q(xk) + R(uk))*(1/(1+ψk'ψk)^2) 
                    for (ψk, xk, uk) in zip(eachcol(ψks), xHistSample, uHistSample)])

indexSort = sortperm(tHistSample)

scatter(tHistSample[indexSort], xHistSample[:,indexSort]')
plot!(tHistSample[indexSort], xHistSample[:,indexSort]')

################################################
############ Saving
################################################
fname = "$dataDir/SRdata.hdf5"
fid = h5open(fname, "w")
fid["Lambda"] = Λ
fid["psiQRks"] = ψQRks 
close(fid)
