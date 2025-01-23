import ArgumentParser
import Foundation
import AppKit
import Tensor

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

        getArgumentAndRun(inputURL: inputURL, testURL: testURL)
    }

    func getArgumentAndRun(inputURL: URL, testURL: URL) {
//        do {
//            let modelURL = inputURL.deletingLastPathComponent().appendingPathComponent("model.safetensors")
//            //try TensorMLPTests().train(inputURL: inputURL, testURL: testURL)
//            try TensorMLPTests().run(modelURL: modelURL, testURL: testURL)
//        } catch {
//            print("Failed with error: \(error)")
//        }
        do {
            let mnistUrl = inputURL.deletingLastPathComponent()
            let dataset = try MNISTLoader.load(from: mnistUrl)

        } catch {
            print(error)
        }
    }
}
