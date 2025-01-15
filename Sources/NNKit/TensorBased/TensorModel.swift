public class TensorModel<Element> where
    Element: BinaryFloatingPoint,
    Element.RawSignificand: FixedWidthInteger
{
    private var layers: [any TensorLayer<Element>] = []

    func addLayer(_ layer: any TensorLayer<Element>) {
        layers.append(layer)
    }

    func parameters() -> [TensorParameter<Element>] {
        layers.flatMap { $0.parameters }
    }
}
