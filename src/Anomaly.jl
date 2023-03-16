module Anomaly

function combine(first, second)
    if first.start < second.start
        return UnitRange(first.start, second.stop)
    else
        return UnitRange(second.start, first.stop)
    end
end

function peak2peak(data::Vector; threshold=100.0)
    indexes = Int64[]
    for i in eachindex(data[1:end-1])
        if abs(data[i] - data[i+1]) > threshold
            push!(indexes, i)
        end
    end

    ranges = UnitRange[0:0]

    for id in indexes
        tmpRange = id-100:id+100
        if length(intersect(ranges[end], tmpRange)) > 0
            ranges[end] = combine(ranges[end], tmpRange)
        else
            push!(ranges, tmpRange)
        end
    end
    return ranges
end

export peak2peak

end # module Anomaly
