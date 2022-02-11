using ProgressMeter

function rkstep4(F, x, δt)
    k1 = δt * F(x)
    k2 = δt * F(x + k1 / 2)
    k3 = δt * F(x + k2 / 2)
    k4 = δt * F(x + k3)
    return x .+ (1 / 6) .* (k1 + 2k2 + 2k3 + k4);
end;

struct hybridSolution
    timeDomain
    arc
end;

struct hybridSolutionCtsArc
    timeDomain
    jumpMarkers
    arc
end;

##### SingleAgent
function hybridSimulator(F, G, flowGuard, jumpGuard,
    x₀, δt,
    maxT)
t = 0
j = 0
x = x₀
discreteTimeDomain = [t,j]
arc = x₀
flow = flowGuard(x)
jump = jumpGuard(x)
progressBar = Progress(convert(Int, ceil(maxT/δt)), 0.5)
while t < maxT && (flow || jump)
    while flow &&  t < maxT
        x = rkstep4(F, x, δt)
        t += δt
        discreteTimeDomain = hcat(discreteTimeDomain, [t,j])
        arc = hcat(arc, x)
        flow = flowGuard(x)
        next!(progressBar)
    end
    jump = jumpGuard(x)
    while jump &&  t < maxT
        x = G(x)
        j += 1
        discreteTimeDomain = hcat(discreteTimeDomain, [t,j])
        arc = hcat(arc, x)
        jump = jumpGuard(x)
    end
    flow = flowGuard(x)        
end
return hybridSolution(discreteTimeDomain, arc)
end;


function hybridSimulatorCtsArc(F, G, flowGuard, jumpGuard,
    x₀, δt,
    maxT)

    continuousTimeDomain = 0:δt:maxT
    totalPoints = size(continuousTimeDomain, 1)
    jumpMarkers = zeros(1, totalPoints)
    arc = zeros(size(x₀,1), totalPoints)

    x = x₀
    iStep = 1
    arc[:, iStep] = x
    t = continuousTimeDomain[iStep]
    j = 0

    flow = flowGuard(x)
    jump = jumpGuard(x)
    progressBar = Progress(totalPoints, 0.5)
    while t < maxT && (flow || jump)
        while flow &&  t < maxT
            x = rkstep4(F, x, δt)
            iStep += 1
            t = continuousTimeDomain[iStep]
            arc[:, iStep] = x
            flow = flowGuard(x)
            next!(progressBar)
        end
        jump = jumpGuard(x)
        while jump &&  t < maxT
            x = G(x)
            j += 1
            jump = jumpGuard(x)
            jumpMarkers[iStep] = j
        end
        flow = flowGuard(x)        
    end

    return hybridSolutionCtsArc(continuousTimeDomain, jumpMarkers, arc)
end;

##### MultiAgent
function hybridSimulatorMa(F, G, flowGuard, jumpGuard,
        x₀, δt,
        maxT)
    t = 0
    j = 0
    x = x₀
    discreteTimeDomain = [t,j]
    arc = x₀
    flow = flowGuard(x)
    jump, triggerIndices = jumpGuard(x)
    progressBar = Progress(convert(Int, ceil(maxT/δt)), 0.5)
    while t < maxT && (flow || jump)
        while flow &&  t < maxT
            x = rkstep4(F, x, δt)
            t += δt
            discreteTimeDomain = hcat(discreteTimeDomain, [t,j])
            arc = hcat(arc, x)
            flow = flowGuard(x)
            next!(progressBar)
        end
        jump, triggerIndices = jumpGuard(x)
        while jump &&  t < maxT
            x = G(x, triggerIndices)
            j += 1
            discreteTimeDomain = hcat(discreteTimeDomain, [t,j])
            arc = hcat(arc, x)
            jump, triggerIndices = jumpGuard(x)
        end
        flow = flowGuard(x)        
    end
    return hybridSolution(discreteTimeDomain, arc)
end;

function hybridSimulatorCtsArcMa(F, G, flowGuard, jumpGuard,
    x₀, δt,
    maxT)

    continuousTimeDomain = 0:δt:maxT
    totalPoints = size(continuousTimeDomain, 1)
    jumpMarkers = zeros(1, totalPoints)
    arc = zeros(size(x₀,1), totalPoints)

    x = x₀
    iStep = 1
    println(size(arc))
    arc[:, iStep] = x
    t = continuousTimeDomain[iStep]
    j = 0

    flow = flowGuard(x)
    jump, triggerIndices = jumpGuard(x)
    progressBar = Progress(totalPoints, 0.5)
    while t < maxT && (flow || jump)
        while flow &&  t < maxT
            x = rkstep4(F, x, δt)
            iStep += 1
            t = continuousTimeDomain[iStep]
            arc[:, iStep] = x
            flow = flowGuard(x)
            next!(progressBar)
        end
        jump, triggerIndices = jumpGuard(x)
        while jump &&  t < maxT
            x = G(x, triggerIndices)
            j += 1
            jump, triggerIndices = jumpGuard(x)
            jumpMarkers[iStep] = j
        end
        flow = flowGuard(x)        
    end

    return hybridSolutionCtsArc(continuousTimeDomain, jumpMarkers, arc)
end

