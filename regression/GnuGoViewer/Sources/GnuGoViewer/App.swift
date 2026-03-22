import SwiftUI

@main
struct GnuGoViewerApp: App {
    @StateObject private var appState: AppState

    init() {
        // Skip argv[0] (program name); also skip "--" if passed by swift run
        let rawArgs = CommandLine.arguments.dropFirst().filter { $0 != "--" }
        let args = Array(rawArgs)

        guard !args.isEmpty else {
            fputs("Usage: GnuGoViewer TEST-FILE:TEST-NUMBER [TEST-FILE:TEST-NUMBER ...]\n", stderr)
            exit(1)
        }

        // Resolve gnugo relative to cwd (regression/GnuGoViewer/ -> ../../interface/gnugo)
        let gnugoPath: String = {
            let candidates = [
                "../../interface/gnugo",
                "../interface/gnugo",
                "./interface/gnugo",
            ]
            for c in candidates {
                if FileManager.default.fileExists(atPath: c) { return c }
            }
            // Fall back; the engine will fail to start but at least the app opens
            fputs("Warning: could not find gnugo binary. Tried: \(candidates.joined(separator: ", "))\n", stderr)
            return candidates[0]
        }()

        let engine = GtpEngine(
            command: [gnugoPath, "--quiet", "--mode", "gtp", "-w", "-t", "-d0x101840"]
        )

        let state = AppState(engine: engine, testcases: args)
        _appState = StateObject(wrappedValue: state)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .frame(minWidth: 900, minHeight: 720)
        }
        .windowStyle(.titleBar)
    }
}
