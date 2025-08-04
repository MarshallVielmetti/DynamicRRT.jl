using Test
using DynamicRRT

@testset "DynamicRRT.jl" begin
    @testset "Core" begin
        include("core/test_spatial_hash_maps.jl")
        include("core/test_rrt_star.jl")
    end

    @testset "Implementations" begin
        include("implementations/test_dynamic_dubins.jl")
    end
end
