import ArgumentParser
import Foundation
import AppKit
import AlgebraKit

@main
struct nnplayground: ParsableCommand {
    @Argument(
        help: "Input file path.",
        completion: .file(),
        transform: URL.init(fileURLWithPath:)
    )
    var inputURL: URL? = nil

    @Argument(
        help: "Input file path.",
        completion: .file(),
        transform: URL.init(fileURLWithPath:)
    )
    var testURL: URL? = nil

    mutating func run() throws {
        guard let inputURL, let testURL else { return }
        Monitor.shared.run { monitor in
            let display1 = Monitor.shared.addDisplay(title: "Test1", size: .zero)
            let display2 = Monitor.shared.addDisplay(title: "Test2", size: .zero)

            let activity = ProcessInfo.processInfo.beginActivity(
                options: .userInitiated,
                reason: "Having some fun here."
            )
            let customQueue = DispatchQueue(
                label: "com.nnpalyground.workQueue",
                qos: .userInitiated
            )
            customQueue.async {
//                do {
//                    try MLPTests().run(url: inputURL, testURL: testURL) { image in
//                        DispatchQueue.main.async {
//                            guard let image else { return }
//                            display1.setImage(image)
//                            display2.setImage(image)
//                        }
//                    }
//                } catch {}

                guard let image = NSImage(data: MachOSectionReader.getEmbeddedData("cats_gs")) else { return }
                DispatchQueue.main.async {
                    display1.setImage(image)
                    var imageMatrix = image.asMatrix()[0]
                    var filter = Matrix.laplacian5x5
                    let fsize = filter.size.cols
                    imageMatrix = im2col(imageMatrix, fsize)
                    filter.reshape(size: Size(1, filter.size.elementCount))
                    var r = filter * imageMatrix
                    r.reshape(size: Size(512 - fsize + 1))
                    r.normalize()
                    r.mapInPlace { $0 = 1.0 / (1.0 + exp(-$0)) }
                    let resultImage = ImageBuilder.buildImage(from: r.padded(25, value: 0), colorTransform: { Viridis.color($0) })
                    display2.setImage(resultImage)
                }
                ProcessInfo.processInfo.endActivity(activity)
            }
        }
    }
}
