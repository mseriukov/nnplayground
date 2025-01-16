// Parametrized extensions are not here yet ): so..
// https://forums.swift.org/t/parameterized-extensions/25563/64

public func stacked<Element>(_ arr: [Tensor<Element>]) -> Tensor<Element> where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    let shape = arr[0].shape
    let batchShape = [arr.count] + shape
    return Tensor<Element>(batchShape, arr.flatMap { $0.storage.data })
}
