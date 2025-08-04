using Test
using DynamicRRT.RRTStar
using DynamicRRT.SpatialHashMaps
using StaticArrays, LinearAlgebra

"""
A mock `AbstractProblem` for testing the core RRT* logic without involving complex dynamics.
This allows for isolated testing of the tree-building, rewiring, and manipulation functions.
The state is a simple 2D point, and steering is a linear interpolation.
"""
struct MockProblem <: RRTStar.AbstractProblem{SVector{2,Float64}}
end

function RRTStar.steer(::MockProblem, x_nearest, x_rand, distance_fraction)
    return x_nearest + distance_fraction * (x_rand - x_nearest), norm(distance_fraction * (x_rand - x_nearest))
end

function RRTStar.collision_free(::MockProblem, x_nearest, x_new)
    return true
end

function RRTStar.spatial_position(::MockProblem, x)
    return x
end

function RRTStar.sample_free(::MockProblem; goal_bias=0.0, hash_map=nothing)
    return @SVector rand(2)
end

function RRTStar.near(problem::MockProblem, hash_map, x_new)
    return 1:length(hash_map)
end

function RRTStar.nearest(problem::MockProblem, hash_map, x_rand)
    return SpatialHashMaps.query_nearest(hash_map, x_rand)
end

@testset "RRTStar" begin

    # Test the fundamental building blocks: Node creation and basic properties.
    @testset "Node & Tree Structure" begin
        # Test RootNode creation
        root = RRTStar.RootNode(@SVector [0.0, 0.0])
        @test root.parent == RRTStar.LinearIndex(0)
        @test root.cost_so_far == 0.0
        @test !root.is_invalid
        @test RRTStar.has_parent(root) # Parent of the root is defined as LinearIndex(0)

        # Test ChildNode creation
        child = RRTStar.ChildNode(SVector{2,Float64}(1.0, 1.0), RRTStar.LinearIndex(1), 1.0)
        @test child.parent == RRTStar.LinearIndex(1)
        @test child.cost_so_far == 1.0
        @test RRTStar.has_parent(child)
    end

    # Test functions that modify the RRT* tree structure.
    @testset "Tree Manipulation" begin
        problem = MockProblem()
        start_state = @SVector [0.0, 0.0]
        hash_map_widths = @SVector [1.0, 1.0]
        solution = RRTStar.setup(problem, start_state, hash_map_widths)

        # Manually add nodes for testing: Root -> A -> B
        node_a = RRTStar.ChildNode(SVector{2,Float64}(1.0, 0.0), RRTStar.LinearIndex(1), 1.0)
        i_a = insert!(solution.hash_map, node_a, SVector{2,Float64}(1.0, 0.0))
        push!(solution.root_node.children, RRTStar.LinearIndex(i_a))

        node_b = RRTStar.ChildNode(SVector{2,Float64}(2.0, 0.0), RRTStar.LinearIndex(i_a), 2.0)
        i_b = insert!(solution.hash_map, node_b, SVector{2,Float64}(2.0, 0.0))
        push!(node_a.children, RRTStar.LinearIndex(i_b))

        @testset "change_parent!" begin
            # Tests changing the parent of a node and verifies that child lists are updated correctly.
            # Change B's parent from A to Root
            RRTStar.change_parent!(problem, solution, RRTStar.LinearIndex(i_b), RRTStar.LinearIndex(1))
            @test solution.hash_map[i_b].parent == RRTStar.LinearIndex(1)
            @test RRTStar.LinearIndex(i_b) in solution.root_node.children
            @test !(RRTStar.LinearIndex(i_b) in solution.hash_map[i_a].children)
        end

        @testset "recalculate_tree!" begin
            # Tests that costs are correctly propagated down the tree after a parent change or cost update.
            # Reset parent for predictability
            solution.hash_map[i_b].parent = RRTStar.LinearIndex(i_a)
            push!(solution.hash_map[i_a].children, RRTStar.LinearIndex(i_b)) # Add it back
            delete!(solution.root_node.children, RRTStar.LinearIndex(i_b)) # Remove from root

            # Update cost of node A and check if B's cost is recalculated
            solution.hash_map[i_a].cost_so_far = 5.0
            RRTStar.recalculate_tree!(problem, solution, RRTStar.LinearIndex(i_a))
            @test solution.hash_map[i_b].cost_so_far ≈ 6.0
        end

        @testset "mark_branch_as_invalid" begin
            # Tests that marking a node as invalid propagates down to all its descendants.
            RRTStar.mark_branch_as_invalid(problem, solution, RRTStar.LinearIndex(i_a))
            @test solution.hash_map[i_a].is_invalid
            @test solution.hash_map[i_b].is_invalid
        end
    end

    # Test the core RRT* algorithm steps.
    @testset "Algorithm Logic" begin
        problem = MockProblem()
        start_state = @SVector [0.0, 0.0]
        hash_map_widths = @SVector [1.0, 1.0]
        solution = RRTStar.setup(problem, start_state, hash_map_widths)

        @testset "rrt_step!" begin
            # Tests that a single step of the RRT* algorithm adds a new node to the tree.
            i_new = RRTStar.rrt_step!(problem, solution)
            @test !isnothing(i_new)
            @test length(solution.hash_map) == 2
        end

        @testset "can_rewire" begin
            # Tests the logic for determining if a node should be rewired to a new parent.
            # Setup a scenario:
            # A
            # | \
            # C--B
            # We test if B should be rewired to be a child of C instead of A.
            node_a = RRTStar.ChildNode(SVector{2,Float64}(0.0, 0.0), RRTStar.LinearIndex(1), 0.0)
            i_a = insert!(solution.hash_map, node_a, SVector{2,Float64}(0.0, 0.0))

            node_b = RRTStar.ChildNode(SVector{2,Float64}(2.0, 0.0), RRTStar.LinearIndex(i_a), 2.0)
            i_b = insert!(solution.hash_map, node_b, SVector{2,Float64}(2.0, 0.0))

            node_c = RRTStar.ChildNode(SVector{2,Float64}(1.0, 1.0), RRTStar.LinearIndex(i_a), sqrt(2.0))
            i_c = insert!(solution.hash_map, node_c, SVector{2,Float64}(1.0, 1.0))

            # Cost from C to B is 1.0, so total cost through C is sqrt(2) + 1 ≈ 2.414
            # Cost from A to B is 2.0. Rewiring should NOT happen.
            @test !RRTStar.can_rewire(problem, solution, RRTStar.LinearIndex(i_b), RRTStar.LinearIndex(i_c))

            # Make it cheaper to go through C by artificially lowering C's cost
            solution.hash_map[i_c].cost_so_far = 0.5
            # Now, cost through C is 0.5 + 1.0 = 1.5, which is less than 2.0. Rewiring SHOULD happen.
            @test RRTStar.can_rewire(problem, solution, RRTStar.LinearIndex(i_b), RRTStar.LinearIndex(i_c))
        end
    end
end

