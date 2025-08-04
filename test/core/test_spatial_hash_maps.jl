using Test
using DynamicRRT.SpatialHashMaps
using StaticArrays

@testset "SpatialHashMaps" begin

    # Test the constructor and basic properties of an empty SpatialHashMap.
    @testset "Initialization" begin
        origin = @SVector [0.0, 0.0]
        widths = @SVector [1.0, 1.0]
        shm = SpatialHashMap{Int}(origin, widths)
        
        # Verify type and initial state
        @test shm isa SpatialHashMap{2,Float64,Int}
        @test length(shm) == 0
        @test element_count(shm) == 0
        @test cell_count(shm) == 0

        # Test that non-positive widths throw an assertion error, as they are invalid.
        @test_throws AssertionError SpatialHashMap{Int}(origin, @SVector [1.0, 0.0])
    end

    # Test the insertion of elements and the resulting state of the hash map.
    @testset "Insertion and State" begin
        origin = @SVector [0.0, 0.0]
        widths = @SVector [1.0, 1.0]
        shm = SpatialHashMap{Int}(origin, widths)

        # Insert first element
        insert!(shm, 1, @SVector [0.5, 0.5])
        @test length(shm) == 1
        @test element_count(shm) == 1
        @test cell_count(shm) == 1

        # Insert second element into a new cell
        insert!(shm, 2, @SVector [1.5, 1.5])
        @test length(shm) == 2
        @test element_count(shm) == 2
        @test cell_count(shm) == 2

        # Insert a third element into an existing cell
        insert!(shm, 3, @SVector [0.6, 0.6])
        @test length(shm) == 3
        @test element_count(shm) == 3
        @test cell_count(shm) == 2 # Should not increase cell count
    end

    # Test querying functions like `query_nearest` and `query_radius`.
    @testset "Querying" begin
        origin = @SVector [0.0, 0.0]
        widths = @SVector [1.0, 1.0]
        shm = SpatialHashMap{Int}(origin, widths)

        p1 = @SVector [0.5, 0.5]
        p2 = @SVector [1.5, 1.5]
        p3 = @SVector [2.5, 2.5]

        insert!(shm, 1, p1)
        insert!(shm, 2, p2)
        insert!(shm, 3, p3)

        @testset "query_nearest" begin
            # Test querying points close to existing elements
            @test query_nearest(shm, @SVector [0.4, 0.4]) == CartesianIndex(1)
            @test query_nearest(shm, @SVector [1.6, 1.6]) == CartesianIndex(2)
            @test query_nearest(shm, @SVector [3.0, 3.0]) == CartesianIndex(3)
            
            # Test querying a point in an empty cell, should return the nearest element
            @test query_nearest(shm, @SVector [-1.0, -1.0]) == CartesianIndex(1)
        end

        # @testset "query_radius" begin
        #     # Tests finding all elements within a given radius of a point.
        #     @test query_radius(shm, @SVector [0.0, 0.0], 0.6) == Set([CartesianIndex(1)])
        #     @test query_radius(shm, @SVector [1.0, 1.0], 0.6) == Set([CartesianIndex(1), CartesianIndex(2)])
        #     @test query_radius(shm, @SVector [2.0, 2.0], 0.6) == Set([CartesianIndex(2), CartesianIndex(3)])
        #     @test isempty(query_radius(shm, @SVector [4.0, 4.0], 0.1))
        # end
    end

    # Test modifying the position of an element in the hash map.
    @testset "Modification" begin
        origin = @SVector [0.0, 0.0]
        widths = @SVector [1.0, 1.0]
        shm = SpatialHashMap{Int}(origin, widths)

        p1 = @SVector [0.5, 0.5]
        p2 = @SVector [1.5, 1.5]

        insert!(shm, 1, p1)
        @test query_nearest(shm, p1) == CartesianIndex(1)

        # Move the element to a new position
        change_position!(shm, 1, p2)

        # The nearest element to the old position should still be the same element index
        @test query_nearest(shm, p1) == CartesianIndex(1)
        # But its internal position should be updated
        @test shm.positions[1] == p2
    end
end
