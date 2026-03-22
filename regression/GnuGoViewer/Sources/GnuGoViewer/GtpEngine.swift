import Foundation

struct GtpResponse {
    var status: String
    var text: String
    var isSuccess: Bool { status == "=" }
    var isFailure: Bool { status == "?" }
}

@Observable
class GtpEngine {
    let commandLine: String
    private let callbackLock = NSLock()
    private var _traceCallback: ((String) -> Void)?
    var traceCallback: ((String) -> Void)? {
        get { callbackLock.lock(); defer { callbackLock.unlock() }; return _traceCallback }
        set { callbackLock.lock(); defer { callbackLock.unlock() }; _traceCallback = newValue }
    }

    private let process: Process
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let stderrPipe = Pipe()
    private var idNumber = 1
    private let lock = NSLock()
    private var crashCallback: (() -> Void)?

    init(command: [String], crashCallback: (() -> Void)? = nil) {
        self.commandLine = command.joined(separator: " ")
        self.crashCallback = crashCallback
        self.process = Process()
        process.executableURL = URL(fileURLWithPath: command[0])
        process.arguments = Array(command.dropFirst())
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try? process.run()

        // Read stderr on a background thread, forwarding to traceCallback
        Thread.detachNewThread { [weak self] in
            guard let self = self else { return }
            let fd = self.stderrPipe.fileHandleForReading.fileDescriptor
            var acc = Data()
            var byte = [UInt8](repeating: 0, count: 1)
            while read(fd, &byte, 1) > 0 {
                if byte[0] == UInt8(ascii: "\n") {
                    if let line = String(data: acc, encoding: .utf8) {
                        let trimmed = line.replacingOccurrences(of: "\r", with: "")
                        if !trimmed.isEmpty { self.traceCallback?(trimmed) }
                    }
                    acc = Data()
                } else {
                    acc.append(byte[0])
                }
            }
        }
    }

    /// Synchronous GTP command. Must NOT be called on the main thread.
    func sendCommand(_ s: String) -> GtpResponse {
        lock.lock()
        defer { lock.unlock() }

        let cmd = "\(idNumber) \(s)\n"
        idNumber += 1

        guard let data = cmd.data(using: .utf8) else {
            return GtpResponse(status: "?", text: "encoding error")
        }
        stdinPipe.fileHandleForWriting.write(data)

        var response = GtpResponse(status: "", text: "")
        var firstLine = true
        var accumulated = Data()

        // Read the response line by line until we hit a blank line (GTP terminator).
        while true {
            // Read one byte at a time from the pipe's file descriptor.
            var byte = [UInt8](repeating: 0, count: 1)
            let fd = stdoutPipe.fileHandleForReading.fileDescriptor
            let n = read(fd, &byte, 1)
            if n <= 0 {
                crashCallback?()
                return GtpResponse(status: "?", text: "engine crashed")
            }
            if byte[0] == UInt8(ascii: "\n") {
                guard var line = String(data: accumulated, encoding: .utf8) else {
                    accumulated = Data(); continue
                }
                accumulated = Data()
                line = line.replacingOccurrences(of: "\r", with: "")

                if firstLine {
                    if line.isEmpty { continue }
                    response.status = String(line.prefix(1))
                    firstLine = false
                    var rest = String(line.dropFirst())
                    if let spaceIdx = rest.firstIndex(of: " ") {
                        let possibleId = String(rest[rest.startIndex..<spaceIdx])
                        if Int(possibleId) != nil {
                            rest = String(rest[rest.index(after: spaceIdx)...])
                        }
                    } else {
                        rest = rest.trimmingCharacters(in: .whitespaces)
                    }
                    response.text = rest
                } else {
                    if line.isEmpty { break }
                    response.text += "\n" + line
                }
            } else {
                accumulated.append(byte[0])
            }
        }
        return response
    }

    func quit() {
        crashCallback = nil
        _ = sendCommand("quit")
    }

    var isRunning: Bool { process.isRunning }
}
