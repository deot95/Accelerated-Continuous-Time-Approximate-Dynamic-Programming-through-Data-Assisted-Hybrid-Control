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
############ Subfolder creation
################################################
fname = "$dataDir/withPESignal.hdf5"
fid = h5open(fname, "r")
timeD = read(fid["timeD"])
arcGradient = read(fid["arcGradient"])
arcHM = read(fid["arcHM"])
close(fid)

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


################################################
############ Auxiliary Functions
################################################
function ϕc(x)
    x1, x2 = x
    return [x1^2; x1*x2; x2^2]
end
dϕc(x) = jacobian(ϕc, x)[1]
dϕc([1;1])
ϕu(x) =  vec(-0.5*Π^(-1)*g(x)'dϕc(x)')

#%%
################################################
############ Data Stacks
################################################

tHist = timeD
xHist = arcGradient[1:n,:]
θcHist = arcGradient[n + 1: n + lc,:]
θuHist = arcGradient[n + lc + 1: n + lc + lu,:]
plot(tHist,θcHist')
plot(tHist,θuHist')

uHist = hcat([reshape(θuk,m,lu)*ϕu(xk) 
                for (θuk, xk) in zip(eachcol(θuHist),eachcol(xHist))]...)

ψ(x, u) =  dϕc(x)*(fKamalapurkar(x) + gKamlapurkar(x)*u)
Ψ(x,u) = ψ(x,u)./(1 .+ ψ(x,u)'ψ(x,u))

#%%
################################################
############ Data Sampling
################################################

nSamples = 100000
Random.seed!(1234)
sampleIndices = sample(1:length(tHist), nSamples, replace=false)
xHistSample = xHist[:,sampleIndices]
tHistSample = tHist[sampleIndices]
uHistSample = uHist[:, sampleIndices]

Λ = zeros(lc, lc)
ψks = zeros(lc, nSamples)
for k in 1:nSamples
    xk,uk = xHistSample[:,k], uHistSample[k]
    ψk = ψ(xk,uk)
    ψks[:,k] = ψk
    Ψk = Ψ(xk,uk)
    Λ +=  Ψk*Ψk'
end
λ̲ = minimum(eigvals(Λ))

ψQRks = sum([ψk*(Q(xk) + R(uk))/(1+ψk'ψk)^2 
                    for (ψk, xk, uk) in zip(eachcol(ψks), xHistSample, uHistSample)])

indexSort = sortperm(tHistSample)

scatter(tHistSample[indexSort], xHistSample[:,indexSort]')
plot!(tHistSample[indexSort], xHistSample[:,indexSort]')

################################################
############ Saving
################################################
fname = "$dataDir/fullSRdataFromPE.hdf5"
fid = h5open(fname, "w")
fid["Lambda"] = Λ
fid["xk"] = xHistSample
fid["uk"] = uHistSample
fid["psik"] =ψks
close(fid)
