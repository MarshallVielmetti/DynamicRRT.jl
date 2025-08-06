using Test
using DynamicRRT
using DynamicRRT.DubinsDynamicPathRRT
using DynamicRRT.RRTStar
using StaticArrays
using Dubins

DubinsDynamicPathRRT = DynamicRRT.DubinsDynamicPathRRT
is_colliding = DubinsDynamicPathRRT.is_colliding

@testset "DubinsDynamicPathRRT" begin

    @testset "Obstacles" begin
        # Static obstacle
        static_obs = DubinsDynamicPathRRT.CircleObstacle(SVector{2,Float64}(1.0, 1.0), 0.5)
        @test is_colliding(static_obs, SVector{3,Float64}(1.0, 1.0, 0.0))
        @test is_colliding(static_obs, SVector{3,Float64}(1.4, 1.0, 0.0))
        @test !is_colliding(static_obs, SVector{3,Float64}(2.0, 2.0, 0.0))

        # Dynamic obstacle
        dyn_obs = DubinsDynamicPathRRT.DynamicCircleObstacle(t -> SVector{2,Float64}(t, t), 0.5, (0.0, 2.0))
        @test is_colliding(dyn_obs, SVector{4,Float64}(1.0, 1.0, 0.0, 1.0))
        @test !is_colliding(dyn_obs, SVector{4,Float64}(1.0, 1.0, 0.0, 3.0)) # Outside time range
        @test !is_colliding(dyn_obs, SVector{4,Float64}(2.0, 1.0, 0.0, 1.0)) # Outside radius
    end

    @testset "Problem and Steering" begin
        domain = (SVector{3,Float64}(0.0, 0.0, 0.0), SVector{3,Float64}(5.0, 5.0, 2pi))
        turning_radius = 1.0
        static_obs = [DubinsDynamicPathRRT.CircleObstacle(SVector{2,Float64}(2.5, 2.5), 0.5)]
        dynamic_obs = [DubinsDynamicPathRRT.DynamicCircleObstacle(t -> SVector{2,Float64}(1.0 + t, 1.0), 0.2, (0.0, 2.0))]
        goal_state = SVector{3,Float64}(4.5, 4.5, 0.0)
        problem = DubinsDynamicPathRRT.DubinsDynamicRRTProblem(domain, turning_radius, static_obs, dynamic_obs, goal_state, true)

        @testset "sample_free" begin
            for _ in 1:10
                q = RRTStar.sample_free(problem)
                @test !is_colliding(problem.static_obstacles, q)
            end
        end

        @testset "steer" begin
            q_near = SVector{4,Float64}(0.0, 0.0, 0.0, 0.0)
            q_rand = SVector{3,Float64}(2.0, 0.0, 0.0)
            q_new, cost = RRTStar.steer(problem, q_near, q_rand, 1.0)

            # Expected path is a straight line of length 2
            @test q_new[1] ≈ 2.0
            @test q_new[2] ≈ 0.0
            @test q_new[3] ≈ 0.0
            @test q_new[4] ≈ 2.0
            @test cost ≈ 2.0
        end

        @testset "collision_free" begin
            # Path through free space
            q1 = SVector{4,Float64}(0.0, 0.0, 0.0, 0.0)
            q2 = SVector{4,Float64}(1.0, 0.0, 0.0, 1.0)
            @test RRTStar.collision_free(problem, q1, q2)

            # Path through static obstacle
            q_static_colliding_start = SVector{4,Float64}(2.5, 2.0, pi / 2, 0.0)
            q_static_colliding_end = SVector{4,Float64}(2.5, 3.0, pi / 2, 1.0)
            @test !RRTStar.collision_free(problem, q_static_colliding_start, q_static_colliding_end)

            # Path through dynamic obstacle
            q_dyn_colliding_start = SVector{4,Float64}(1.5, 0.8, 0.0, 0.5)
            q_dyn_colliding_end = SVector{4,Float64}(2.5, 0.8, 0.0, 1.5)

            # at t=1.0, obstacle is at [2.0, 1.0], path is at [2.0, 0.8] -> collision
            @test !RRTStar.collision_free(problem, q_dyn_colliding_start, q_dyn_colliding_end)
        end

        @testset "goal_reachable" begin
            # This needs a solution object.
            start_state = SVector{4,Float64}(0.0, 0.0, 0.0, 0.0)
            hash_map_widths = SVector{2,Float64}(1.0, 1.0)
            solution = RRTStar.setup(problem, start_state, hash_map_widths)

            # Node far from goal
            node_far = RRTStar.ChildNode(SVector{4,Float64}(1.0, 1.0, 0.0, 1.0), RRTStar.LinearIndex(1), 1.0)
            i_far = insert!(solution.hash_map, node_far, RRTStar.spatial_position(problem, node_far.state))
            @test !RRTStar.goal_reachable(problem, solution.hash_map, i_far)

            # Node near goal
            node_near = RRTStar.ChildNode(SVector{4,Float64}(4.5, 4.5, 0.0, 5.0), RRTStar.LinearIndex(1), 5.0)
            i_near = insert!(solution.hash_map, node_near, RRTStar.spatial_position(problem, node_near.state))
            @test RRTStar.goal_reachable(problem, solution.hash_map, i_near)
        end
    end
    @testset "Path Extraction" begin
        domain = (SVector{3,Float64}(0.0, 0.0, 0.0), SVector{3,Float64}(5.0, 5.0, 2pi))
        turning_radius = 1.0
        problem = DubinsDynamicPathRRT.DubinsDynamicRRTProblem(domain, turning_radius, [], [], SVector{3,Float64}(3.0, 0.0, 0.0), true)
        start_state = SVector{4,Float64}(0.0, 0.0, 0.0, 0.0)
        hash_map_widths = @SVector [1.0, 1.0]
        solution = RRTStar.setup(problem, start_state, hash_map_widths)

        # Manually construct a simple path: Root -> A -> B -> Goal
        node_a = RRTStar.ChildNode(SVector{4,Float64}(1.0, 0.0, 0.0, 1.0), RRTStar.LinearIndex(1), 1.0)
        i_a = insert!(solution.hash_map, node_a, RRTStar.spatial_position(problem, node_a.state))
        push!(solution.root_node.children, RRTStar.LinearIndex(i_a))

        node_b = RRTStar.ChildNode(SVector{4,Float64}(2.0, 0.0, 0.0, 2.0), RRTStar.LinearIndex(i_a), 2.0)
        i_b = insert!(solution.hash_map, node_b, RRTStar.spatial_position(problem, node_b.state))
        push!(node_a.children, RRTStar.LinearIndex(i_b))

        node_goal = RRTStar.ChildNode(SVector{4,Float64}(3.0, 0.0, 0.0, 3.0), RRTStar.LinearIndex(i_b), 3.0)
        i_goal = insert!(solution.hash_map, node_goal, RRTStar.spatial_position(problem, node_goal.state))
        push!(node_b.children, RRTStar.LinearIndex(i_goal))

        @testset "extract_path" begin
            path = RRTStar.extract_path(problem, solution, RRTStar.LinearIndex(i_goal))
            @test path[1] == RRTStar.path_pose(problem, start_state)
            @test path[2] == RRTStar.path_pose(problem, node_a.state)
            @test path[3] == RRTStar.path_pose(problem, node_b.state)
            @test path[4] == RRTStar.path_pose(problem, node_goal.state)
        end

        @testset "best_path!" begin
            RRTStar.best_path!(problem, solution)
            @test solution.status == RRTStar.GoalReachable
            @test solution.best_path_cost == 3.0
        end
    end
end