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
        guard
            let inputURL,
            let testURL
        else { return }
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

//                do {
//                    try ConvTests().process(input: inputURL) { image in
//                        DispatchQueue.main.async {
//                            guard let image else { return }
//                            display1.setImage(image)
//                        }
//                    }
//                } catch {}

                guard let image = NSImage(data: MachOSectionReader.getEmbeddedData("cats")) else { return }
                DispatchQueue.main.async {
                    display1.setImage(image)
                    var imageMatricies = image.asMatrix()

                    var filter = Matrix.gaussian5x5
                    let fsize = filter.size.cols
                    filter.reshape(size: Size(1, filter.size.elementCount))
                    var r = imageMatricies.map {
                        filter * im2col($0, fsize)
                    }
                    r[0].reshape(size: Size(512 - fsize + 1))
                    r[1].reshape(size: Size(512 - fsize + 1))
                    r[2].reshape(size: Size(512 - fsize + 1))
                    r[0].scaleToUnitInterval()
                    r[1].scaleToUnitInterval()
                    r[2].scaleToUnitInterval()
                    let resultImage = ImageBuilder.buildImage(from: r)
                    display2.setImage(resultImage)
                }
                ProcessInfo.processInfo.endActivity(activity)
            }
        }
    }
}
