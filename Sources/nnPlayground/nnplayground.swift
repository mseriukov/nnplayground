import ArgumentParser

@main
struct nnplayground: ParsableCommand {
    mutating func run() throws {
        MLPTests().run()
    }
}

