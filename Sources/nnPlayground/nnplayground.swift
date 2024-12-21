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
        //guard let inputURL, let testURL else { return }
        Monitor.shared.run { monitor in
            let display1 = Monitor.shared.addDisplay(title: "Test1", size: .zero)
            let display2 = Monitor.shared.addDisplay(title: "Test2", size: .zero)
            DispatchQueue.global(qos: .userInitiated).async {
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
                    let fsize = filter.cols
                    imageMatrix = im2col(imageMatrix, fsize)
                    filter.reshape(rows: 1, cols: fsize * fsize)
                    var r = filter * imageMatrix
                    r.reshape(rows: 512 - fsize + 1, cols: 512 - fsize + 1)
                    r.normalize()                    
                    let resultImage = ImageBuilder.buildImage(from: r)
                    display2.setImage(resultImage)
                }
            }
        }
    }
}
