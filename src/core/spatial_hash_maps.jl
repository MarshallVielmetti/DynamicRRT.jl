module SpatialHashMaps

using StaticArrays, LinearAlgebra

export SpatialHashMap
export query_radius, query_nearest, change_position!
export cell_count, element_count

struct SpatialHashMap{DIM,F,T}
    origin::SVector{DIM,F}
    widths::SVector{DIM,F}
    elements::Vector{T}
    positions::Vector{SVector{DIM,F}} # stores the spatial position of each element
    hash_map::Dict{CartesianIndex{DIM},Set{CartesianIndex{1}}} # stores a set from the pos_idx to the lin_idx
end


# empty constructor
function SpatialHashMap{T}(origin::SVector{DIM,F}, widths::SVector{DIM,F}) where {DIM,F,T}

    @assert minimum(widths) > 0 "all widths must be positive"

    # empty list of elements
    elements = T[]

    # empty list of positions
    positions = SVector{DIM,F}[]

    # empty hash map
    hash_map = Dict{CartesianIndex{DIM},Set{CartesianIndex{1}}}()

    return SpatialHashMap(origin, widths, elements, positions, hash_map)
end


# indexing
function get_index(shm::SpatialHashMap{DIM,F,T}, pos::SVector{DIM,F}) where {DIM,F,T}
    o = shm.origin
    w = shm.widths
    ind = CartesianIndex(ntuple(i -> Int(fld(pos[i] - o[i], w[i])), DIM))
    return ind
end

# query methods
function get_cell(shm::SpatialHashMap{DIM,F,T}, index::CartesianIndex{DIM}) where {DIM,F,T}
    # get element indices at this cell
    # return an empty set if the index is not defined yet

    if haskey(shm.hash_map, index)
        return shm.hash_map[index]
    else
        return empty_set = Set{CartesianIndex{1}}()
    end
    # return get(shm.hash_map, index, empty_set)
end

function get_cell(shm::SpatialHashMap{DIM,F,T}, pos::SVector{DIM,F}) where {DIM,F,T}
    idx = get_index(shm, pos)
    return get_cell(shm, idx)
end

function neighborhood_indices(shm::SpatialHashMap{DIM,F}, pos::SVector{DIM,F}, radius::F) where {DIM,F}
    r = radius * (@SVector ones(DIM))
    return neighborhood_indices(shm, pos, r)
end

function neighborhood_indices(shm::SpatialHashMap{DIM,F}, pos::SVector{DIM,F}, radius::SVector{DIM,F}) where {DIM,F}
    # TODO(dev): consider adding periodic dimensions
    min_idx = get_index(shm, pos - radius)
    max_idx = get_index(shm, pos + radius)
    return min_idx:max_idx
end

function query_radius(shm::SpatialHashMap{DIM,F,T}, pos::SVector{DIM,F}, radius) where {DIM,F,T}
    # todo: add in a strict flag

    pos_inds = neighborhood_indices(shm, pos, radius)
    # lin_inds = mapreduce(i -> get_cell(shm, i), union, pos_inds)

    # use haskey to speed it up
    lin_inds = Set{CartesianIndex{1}}()
    for cell_ind in pos_inds
        if haskey(shm.hash_map, cell_ind)
            union!(lin_inds, shm.hash_map[cell_ind])
        end
    end

    return lin_inds
end

function query_nearest_key(shm::SpatialHashMap{DIM,F,T}, pos_idx::CartesianIndex{DIM}) where {DIM,F,T}

    best_key = nothing
    best_d = Inf
    for k in keys(shm.hash_map)
        d = norm(Tuple(k - pos_idx))
        if d < best_d
            best_d = d
            best_key = k
        end
    end

    return best_key
end

function query_nearest(shm::SpatialHashMap{DIM,F,T}, pos::SVector{DIM,F}) where {DIM,F,T}
    # returns the index to the nearest element
    @assert length(shm.positions) > 0 "there must be atleast one element in the spatial hash map"

    pos_idx = get_index(shm, pos)
    nearest_key = query_nearest_key(shm, pos_idx)

    # get the maximum radius to consider
    r = maximum(ntuple(i -> abs(nearest_key[i] - pos_idx[i]), DIM))

    # get all pos_inds within this radius
    cell_inds = CartesianIndices(ntuple(i -> (pos_idx[i]-r):(pos_idx[i]+r), DIM))


    # loop through all elements in cell_inds
    best_ind = nothing
    best_d = Inf
    for cell_ind in cell_inds
        if haskey(shm.hash_map, cell_ind)
            for lin_ind in shm.hash_map[cell_ind]
                d = norm(shm.positions[lin_ind] - pos)
                if d < best_d
                    best_ind = lin_ind
                    best_d = d
                end
            end
        end
    end

    @assert !isnothing(best_ind) "something went wrong"
    return best_ind

end


# insertion methods
function Base.insert!(shm::SpatialHashMap{DIM,F,T}, el::T, pos::SVector{DIM,F}) where {DIM,F,T}

    # push it to the elements vector and the spatial position vector
    push!(shm.elements, el)
    push!(shm.positions, pos)

    # get the linear and cartesian indices
    lin_idx = length(shm.elements)
    pos_idx = get_index(shm, pos)

    # initialize the set if it doesnt already exist
    if !haskey(shm.hash_map, pos_idx)
        shm.hash_map[pos_idx] = Set{CartesianIndex{1}}()
    end

    # push it to the dictionary at that cell
    push!(shm.hash_map[pos_idx], CartesianIndex(lin_idx))

    # return the linear index that we added the thing at
    return lin_idx
end


# modify method
function change_position!(shm::SpatialHashMap{DIM,F,T}, lin_idx, new_pos::SVector{DIM,F}) where {DIM,F,T}

    # get old position
    old_pos = shm.positions[lin_idx]
    old_pos_idx = get_index(shm, old_pos)

    # remove old pos from hash_map
    delete!(shm.hash_map[old_pos_idx], old_pos_idx)
    if length(shm.hash_map[old_pos_idx]) == 0
        # delete this from the dictionary
        delete!(shm.hash_map, old_pos_idx) #todo(dev): check if this actually works
    end

    # update the position vector
    shm.positions[lin_idx] = new_pos

    # add into new hash_map pos
    new_pos_idx = get_index(shm, new_pos)
    if !haskey(shm.hash_map, new_pos_idx)
        shm.hash_map[new_pos_idx] = Set{CartesianIndex{1}}()
    end
    push!(shm.hash_map[new_pos_idx], CartesianIndex(lin_idx))

end


# # a few utilities
function element_count(shm::SpatialHashMap)
    return length(shm.elements)
end

function cell_count(shm::SpatialHashMap)
    return length(keys(shm.hash_map))
end


# # try iterating utils
Base.iterate(shm::SpatialHashMap) = iterate(shm.elements)
Base.iterate(shm::SpatialHashMap, state) = iterate(shm.elements, state)
Base.IteratorSize(shm::SpatialHashMap) = Base.HasLength()
Base.length(shm::SpatialHashMap) = length(shm.elements)
Base.eltype(shm::SpatialHashMap{DIM,F,T}) where {DIM,F,T} = T
Base.getindex(shm::SpatialHashMap, inds...) = getindex(shm.elements, inds...)

# pretty printing

# one-line print:
function Base.show(io::IO, shm::SpatialHashMap{DIM,F,T}) where {DIM,F,T}
    print(io, typeof(shm))
end

# multi-line print:
function Base.show(io::IO, ::MIME"text/plain", shm::SpatialHashMap{DIM,F,T}) where {DIM,F,T}
    print(io, "SpatialHashMap{$(DIM), $(F), $(T)}\n")
    print(io, " - origin   : $(shm.origin)\n")
    print(io, " - widths   : $(shm.widths)\n")
    print(io, " - cells    : $(cell_count(shm))\n")
    print(io, " - elements : $(element_count(shm))\n")
end



# # spiral shell iteration utilities
# struct SpiralShellIterator{D}
#     center::CartesianIndex{D}
#     max_radius::Union{Nothing, Int}  # Optional limit
# end

# SpiralShellIterator(center::CartesianIndex{D}) where {D} = SpiralShellIterator{D}(center, nothing)
# SpiralShellIterator(center::CartesianIndex{D}, max_radius::Int) where {D} = SpiralShellIterator{D}(center, max_radius)

# Base.iterate(iter::SpiralShellIterator{D}) where {D} = iterate(iter, 0)
# function Base.iterate(iter::SpiralShellIterator{D}, radius::Int) where {D}
#     if !isnothing(iter.max_radius) && radius > iter.max_radius
#         return nothing
#     end

#     # range = ntuple(i -> (iter.center[i]-radius):(iter.center[i]+radius), D)
#     # shell = CartesianIndices(range)
#     # return shell, radius + 1

#     range = ntuple(i -> (-radius):(radius), D)
#     shell = CartesianIndices(range)
#     result = CartesianIndex[]
#     sizehint!(result, max(1, (2*radius+1)^D - (2*(radius-1)+1)^2))
#     for offset in shell
#         if maximum(abs, Tuple(offset)) == radius
#             push!(result, iter.center + offset)
#         end
#     end

#     return result, radius + 1

# end



end