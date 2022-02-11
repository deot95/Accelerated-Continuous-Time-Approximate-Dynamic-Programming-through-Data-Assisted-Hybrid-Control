using LinearAlgebra

function FPlant(x,u,f,g)
    fx = f(x)
    gx = g(x)
    ẋ = fx + gx.*u
    return ẋ
end

function FGradientCritic(y, x, u, lc, kc, ρi, ρd, dϕc, fModel, gModel, Q, R, ∇d)
	#=
		kc ∈ ℜⁿ: gain vector
		fModel: Free flow model
        gModel: Input matrix function model
		∇d: Λ + ∑ (Q(xₖ) + R(uₖ))/(1+ψ(xₖ)ᵀ\psi(xₖ))
	=#
	########################
	### State selection
	########################		
	θc = y[1:lc]

    ########################
	### Critic Error Gradient
	########################
    ψx =  dϕc(x)*(fModel(x) + gModel(x).*u)
    nψx = (1 .+ ψx'ψx) #Normalization Term

    Ψx = ψx/nψx
    Qx = Q(x)
    Ru = R(u)

    ∇i = Ψx*Ψx'θc + ψx*(Qx + Ru)/nψx^2
    ∇e = ρi*∇i + ρd*∇d(θc)

	########################
	### Flows
	########################
	θ̇c = -kc*∇e
	return θ̇c
end

function FHMCritic(y, x, u, lc, kc, ρi, ρd, dϕc, fModel, gModel, Q, R, ∇d)
	#=
		kc ∈ ℜⁿ: gain vector
		fModel: Free flow model
        gModel: Input matrix function model
		∇d: Λ + ∑ (Q(xₖ) + R(uₖ))/(1+ψ(xₖ)ᵀ\psi(xₖ))
	=#
	########################
	### State selection
	########################		
	θc = y[1:lc] ### Position variable: θc ∈ ℜⁿ
    p = y[lc+ 1:lc+ lc] ### Momentum variable: p ∈ ℜⁿ
    τ = y[end] ### Vector of timers: τ ∈ ℜᴺ

    ########################
	### Critic Error Gradient
	########################
    ψx =  dϕc(x)*(fModel(x) + gModel(x).*u)
    nψx = (1 .+ ψx'ψx) #Normalization Term

    Ψx = ψx/nψx
    Qx = Q(x)
    Ru = R(u)

    ∇i = Ψx*Ψx'θc + ψx*(Qx + Ru)/nψx^2
    ∇e = ρi*∇i + ρd*∇d(θc)

	########################
	### Flows
	########################
	θ̇c = 2*(p - θc)/τ
	ṗ = -2kc*τ*∇e
	τ̇ = 1/2
	return [θ̇c; ṗ; τ̇]
end


function GHMCritic(y, T₀, α)
	#=
		kc ∈ ℜⁿ: gain vector
		fModel: Free flow model
        gModel: Input matrix function model
		∇d: Λ + ∑ (Q(xₖ) + R(uₖ))/(1+ψ(xₖ)ᵀ\psi(xₖ))
	=#
	########################
	### State selection
	########################		
	θc = y[1:lc] ### Position variable: θc ∈ ℜⁿ
    p = y[lc+ 1:lc+ lc] ### Momentum variable: p ∈ ℜⁿ
    τ = y[end] ### Vector of timers: τ ∈ ℜᴺ

    ########################
	### Jump Logic
	########################
	### Position and momentum variables
	θc⁺ = θc
	p⁺ = α .* p + (1 - α) .* θc
    τ⁺ = T₀
    return [θc⁺; p⁺; τ⁺]
end


function FActor(θu, x, θc, ku, ϕu, dϕc, gModel, Π)
    """
    θu: Actor Matrix Parameters
    x: Plant state
    θc: Critic Vector Parameters
    """
    ########################
	### Actor Error Gradient
	########################
    ϕux = ϕu(x)
    dϕcx = dϕc(x)
    gx = gModel(x)
    ∇ε = (θu*ϕux .+ 0.5*Π^(-1)*gx'dϕcx'θc)*ϕux'

    ########################
	### Flow
	########################
    θ̇u = -ku*∇ε
    return θ̇u
end

function FClosedLoopGradientCritic(z, n, m, lc, lu, f, g, kc, ρi, ρd,
    dϕc, fModel, gModel, Q, R, ∇d, ku, ϕu, Π, pe)

    ########################
    ### State selection
    ########################	                         

    x = z[1:n]
    y = z[n+ 1:n+ lc]
    θc = y[1:lc]
    θu = reshape(z[n+lc+ 1:n+lc+ m*lu], (m,lu))
    t = z[end]

    ########################
    ### Control Action
    ########################    
    ϕux = ϕu(x)
    ûx = θu*ϕux

    ########################
    ### Flows
    ########################

    ẋ = FPlant(x, ûx .+ pe(t), f, g)
    ẏ = FGradientCritic(y, x, ûx, lc, kc, ρi, ρd, dϕc, fModel, gModel, Q, R, ∇d)
    θ̇u = FActor(θu, x, θc, ku, ϕu, dϕc, gModel, Π)

    vecθ̇u = vec(θ̇u)
    ṫ = 1

    return [ẋ; ẏ; vecθ̇u; ṫ]
end


function FClosedLoopHMCritic(z, n, m, lc, lu, f, g, kc, ρi, ρd,
                         dϕc, fModel, gModel, Q, R, ∇d, ku, ϕu, Π, pe)
    
	########################
	### State selection
	########################	                         
    x = z[1:n]
    y = z[n+ 1:n+ 2lc+1] #[θ, p, τ]∈ℜ²ˡᶜ⁺¹
    θc = y[1:lc]
    θu = reshape(z[n+2lc+1+ 1 : n+2lc+1+ m*lu], (m,lu))
    t = z[end]

    ########################
	### Control Action
	########################    
    ϕux = ϕu(x)
    ûx = θu*ϕux

    ########################
	### Flows
	########################
    ẋ = FPlant(x, ûx .+ pe(t), f, g)
    ẏ = FHMCritic(y,x,ûx,lc,kc,ρi,ρd, dϕc, fModel, gModel, Q, R, ∇d)
    θ̇u = FActor(θu, x, θc, ku, ϕu, dϕc, gModel, Π)

    vecθ̇u = vec(θ̇u)
    ṫ =  1

    return [ẋ; ẏ; vecθ̇u; ṫ]
end

function FActorAlt(θu, x, θc, ku, α₁, α₂, dϕc, gModel, Πu)
    """
    θu: Actor Matrix Parameters
    x: Plant state
    θc: Critic Vector Parameters
    """
    ########################
	### Actor Error Gradient
	########################
    gx = gModel(x)
    dϕcx = dϕc(x)
    invΠu = Πu^(-1)
    ωx = -0.5*dϕcx*gx*invΠu
    
    Ω̃x = ωx*ωx'/(1+tr(ωx'ωx))
    Ωx = α₁*Ω̃x + α₂*I

    ∇ε = Ωx*(θu-θc)

    ########################
	### Flow
	########################
    θ̇u = -ku*∇ε
    return θ̇u
end

function FClosedLoopGradientCriticAlt(z, n, lc, f, g, kc, ρi, ρd,
    dϕc, fModel, gModel, Q, R, ∇d, ku, α₁, α₂, Πu, pe)

    ########################
    ### State selection
    ########################	                         
    x = z[1:n]
    y = z[n+ 1:n+ lc]
    θc = y[1:lc]
    θu = z[n+lc+ 1:n+2lc]
    t = z[end]

    ########################
    ### Control Action
    ########################
    gx = gModel(x)
    dϕcx = dϕc(x)
    invΠu = Πu^(-1)
    ωx = -0.5*dϕcx*gx*invΠu    
    ûx = ωx'θu

    ########################
    ### Flows
    ########################

    ẋ = FPlant(x, ûx .+ pe(t), f, g)
    ẏ = FGradientCritic(y, x, ûx, lc, kc, ρi, ρd, dϕc, fModel, gModel, Q, R, ∇d)
    θ̇u = FActorAlt(θu, x, θc, ku, α₁, α₂, dϕc, gModel, Πu)

    vecθ̇u = vec(θ̇u)
    ṫ = 1

    return [ẋ; ẏ; vecθ̇u; ṫ]
end

function FClosedLoopHMCriticAlt(z, n, lc, f, g, kc, ρi, ρd,
    dϕc, fModel, gModel, Q, R, ∇d, ku, α₁, α₂, Πu, pe)

    x = z[1:n]
    y = z[n+ 1:n+ 2lc+1] #[θ, p, τ]∈ℜ²ˡᶜ⁺¹
    θc = y[1:lc]
    θu = z[n+2lc+1+ 1 : n+2lc+1+ lc]
    t = z[end]

    ########################
    ### Control Action
    ########################
    gx = gModel(x)
    dϕcx = dϕc(x)
    invΠu = Πu^(-1)
    ωx = -0.5*dϕcx*gx*invΠu    
    ûx = ωx'θu

    ########################
    ### Flows
    ########################
    ẋ = FPlant(x, ûx .+ pe(t), f, g)
    ẏ = FHMCritic(y, x, ûx, lc, kc, ρi, ρd, dϕc, fModel, gModel, Q, R, ∇d)
    θ̇u = FActorAlt(θu, x, θc, ku, α₁, α₂, dϕc, gModel, Πu)

    vecθ̇u = vec(θ̇u)
    ṫ =  1

    return [ẋ; ẏ; vecθ̇u; ṫ]
end


function GClosedLoopHMCritic(z, n, m, lc, lu, α)
    ########################
	### State selection
	########################	                         
    x = z[1:n]
    y = z[n+ 1: n+ 2lc+1]
    vecθu = z[n+2lc+1+ 1: n+2lc+1+ m*lu]
    t = z[end]

    x⁺ = x
    y⁺ = GHMCritic(y, T₀, α)
    vecθu⁺ = vecθu
    t⁺ = t

    return [x⁺; y⁺; vecθu⁺; t⁺]
end

function flowGuardRestarting(z, n, lc, T₀, T)
	########################
	### State selection
	########################
    y = z[n+1: n+2lc+1]
    τ = y[end]
    return (τ >= T₀) && (τ <= T)
end

function jumpGuardRestarting(z, n, lc, T₀, T)
	########################
	### State selection
	########################
    y = z[n+1: n+2lc+1]
    τ = y[end]
    
    return τ >= T
end

##############################
### In place jumps
##############################

function GClosedLoopHMCritic!(integrator, n, lc, α, T₀)
    ########################
	### State selection
	########################	                         
    θc = integrator.u[n + 1: n+lc]
    p = integrator.u[n + lc + 1: n + 2lc]
    integrator.u[n + lc + 1: n + 2lc ] = α.*p + (1-α).*θc
    integrator.u[n+2lc+1] = T₀
end