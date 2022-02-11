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
############ Auxiliary Functions
################################################
function ϕc(x)
    x1, x2 = x
    return [x1^2; x1*x2; x2^2]
end
dϕc(x) = jacobian(ϕc, x)[1]
dϕc([1;1])

u(x) =  -(cos(2x[1]) + 2)*x[2]
ψ(x, u) =  dϕc(x)*(fKamalapurkar(x) + gKamlapurkar(x)*u)

#%%
################################################
############ Data Sampling
################################################
xRange = range(-3, 3, 4)
xSample = Iterators.product(xRange, xRange)
xSampleVec = hcat(vec([collect(x) for x in xSample])...)
uSampleVec= u.(eachcol(xSampleVec))

Λ = zeros(lc, lc)
ψQRks = zeros(lc, 1)
for (k,(xk,uk)) in enumerate(zip(eachcol(xSampleVec), uSampleVec))
    ψk = ψ(xk,uk)
    nψk = 1 + ψk'ψk
    Ψk = ψk/nψk
    Λ +=  Ψk*Ψk'
    ψQRks += ψk*(Q(xk) + R(uk))*(1/(1+ψk'ψk)^2) 
end
λ̲ = minimum(eigvals(Λ))

################################################
############ Saving
################################################
fname = "$dataDir/SRDataAlaKamalapurkar.hdf5"
fid = h5open(fname, "w")
fid["Lambda"] = Λ
fid["psiQRks"] = ψQRks 
close(fid)
