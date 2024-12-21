//
//  MachOSectionReader.swift
//  nnplayground
//
//  Created by Mikhail Seriukov on 21/12/2024.
//

import Foundation

struct MachOSectionReader {
    /// Get resource embedded in the executable binary.
    ///
    /// The resource needs to be embedded by the linker using the flag
    /// `-sectcreate __TEXT [name] [path to file]`.
    static func getEmbeddedData(_ name: String) -> Data {
        // All of this because `_mh_execute_header` is a `let` instead of a `var`,
        // and `getsectiondata` does not accept copies of this value.
        // See https://stackoverflow.com/a/49438718/400056
        guard let handle = dlopen(nil, RTLD_LAZY) else {
            fatalError("Cannot get handle to executable")
        }
        defer { dlclose(handle) }

        guard let ptr = dlsym(handle, MH_EXECUTE_SYM) else {
            fatalError("Cannot get symbol '\(MH_EXECUTE_SYM)")
        }
        let header = ptr.assumingMemoryBound(to: mach_header_64.self)
        var size: UInt = 0
        guard let rawData = getsectiondata(header, "__TEXT", name, &size) else {
            fatalError("Cannot find resource '\(name)'")
        }
        return Data(bytes: rawData, count: Int(size))
    }
}
