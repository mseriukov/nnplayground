
extension Tensor {
    public func reduce(
        initialValue: Double,
        reduceFunction: (Double, Double) -> Double
    ) -> Double {
        var result = initialValue
        self.forEachIndex { index in
            result = reduceFunction(result, self[index])
        }
        return result
    }

    public func reduce(
        alongAxis axis: Int,
        keepDims: Bool = false,
        reduceFunction: (Double, Double) -> Double
    ) -> Tensor {
        // Ensure axis is within bounds
        guard axis >= 0 && axis < shape.count else {
            fatalError("Axis out of bounds for tensor shape \(shape).")
        }

        // Compute the new shape
        var resultShape = shape
        if keepDims {
            resultShape[axis] = 1 // Retain the dimension as size 1
        } else {
            resultShape.remove(at: axis) // Remove the dimension
        }

        // Create a result tensor with the new shape
        var result = Self(zeros: resultShape)

        // Iterate through all indices of the result tensor
        result.forEachIndex { resultIndex in
            // Map resultIndex back to the original tensor indices
            var originalIndex = Array(resultIndex.prefix(axis))
            if keepDims {
                originalIndex.append(0) // Placeholder for the reducing axis
                originalIndex.append(contentsOf: resultIndex.suffix(resultIndex.count - axis - 1))
            } else {
                originalIndex.append(0) // Placeholder for reducing axis
                originalIndex.append(contentsOf: resultIndex.suffix(resultIndex.count - axis))
            }

            // Apply reduction along the specified axis
            var accumulatedValue = self[originalIndex]
            for i in 1..<shape[axis] {
                originalIndex[axis] = i
                accumulatedValue = reduceFunction(accumulatedValue, self[originalIndex])
            }

            result[resultIndex] = accumulatedValue
        }

        return result
    }
}

extension Tensor {
    public func sum(alongAxis axis: Int, keepDims: Bool = false) -> Self {
        reduce(alongAxis: axis, keepDims: keepDims, reduceFunction: +)
    }

    public func product(alongAxis axis: Int, keepDims: Bool = false) -> Self {
        reduce(alongAxis: axis, keepDims: keepDims, reduceFunction: *)
    }

    public func min(alongAxis axis: Int, keepDims: Bool = false) -> Self {
        reduce(alongAxis: axis, keepDims: keepDims, reduceFunction: Swift.min)
    }

    public func max(alongAxis axis: Int, keepDims: Bool = false) -> Self {
        reduce(alongAxis: axis, keepDims: keepDims, reduceFunction: Swift.max)
    }

    public func sum() -> Self {
        let value = reduce(initialValue: 0, reduceFunction: +)
        return Self([1], [value])
    }

    public func max() -> Self {
        let value = reduce(initialValue: Double.leastNonzeroMagnitude, reduceFunction: Swift.max)
        return Self([1], [value])
    }
    
    public func min() -> Self {
        let value = reduce(initialValue: Double.greatestFiniteMagnitude, reduceFunction: Swift.min)
        return Self([1], [value])
    }
}
