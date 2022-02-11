#=
Online actor–critic algorithm to solve the
continuous-time infinite horizon optimal 
control problem-Kyriakos 2010 Automatica. Plant.
=#
function fAircraft(x)
    A = [-1.01887  0.90506 -0.00215 ;
         0.82225 -1.07741  -0.17555 ;
            0        0         -1   ]
    return A*x
end

function gAircraft(x)
    B = reshape([0; 0; 1], (3,1))
    return B
end


#=
Van Der Pol Asymptotically Stable Adaptive Optimal Control
Algorithm with Saturating Actuators and Relaxed Persistence
of Excitation IEE Transactions on Neural Networks
=#
function fVanDerPol(x)
    x1, x2 = x
    ẋ1 = x2
    ẋ2 = -x1 - 0.5x2*(1-x1^2)
    return [ẋ1; ẋ2]
end


function gVanDerPol(x)
    x1, x2 = x
    return [0; x1]
end



#=
Nonlinear Example from Kamalapurkar and 2010 Kyriakos
=#
function fKamalapurkar(x)
    x1, x2 = x
    ẋ1 = -x1 + x2
    ẋ2 = -0.5x1 - 0.5x2*(1-(cos(2x1) + 2)^2)
    return [ẋ1; ẋ2]
end


function gKamalapurkar(x)
    x1, x2 = x
    return [0; cos(2x1) + 2]
end
