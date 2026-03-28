import SwiftUI
import AppKit

@main
struct GnuGoViewerApp: App {
    @State private var appState: AppState

    init() {
        // Allow the window to appear when launched from the terminal (not a .app bundle).
        NSApplication.shared.setActivationPolicy(.regular)

        // Skip argv[0] (program name); also skip "--" if passed by swift run
        let rawArgs = CommandLine.arguments.dropFirst().filter { $0 != "--" }
        let args = Array(rawArgs)

        guard !args.isEmpty else {
            fputs("Usage: GnuGoViewer TEST-FILE:TEST-NUMBER [TEST-FILE:TEST-NUMBER ...]\n", stderr)
            exit(1)
        }

        // Resolve gnugo relative to cwd (regression/GnuGoViewer/ -> ../../interface/gnugo)
        let gnugoPath: String = {
            let cwd = FileManager.default.currentDirectoryPath
            let candidates = [
                "../../interface/gnugo",
                "../interface/gnugo",
                "./interface/gnugo",
            ]
            for c in candidates {
                if FileManager.default.fileExists(atPath: c) {
                    // Resolve to absolute path — URL(fileURLWithPath:) drops ../ components.
                    return URL(fileURLWithPath: cwd + "/" + c).standardized.path
                }
            }
            // Fall back; the engine will fail to start but at least the app opens
            fputs("Warning: could not find gnugo binary. Tried: \(candidates.joined(separator: ", "))\n", stderr)
            return candidates[0]
        }()

        let engine = GtpEngine(
            command: [gnugoPath, "--quiet", "--mode", "gtp", "-w", "-t", "-d0x101840"]
        )

        let state = AppState(engine: engine, testcases: args)
        _appState = State(initialValue: state)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(appState)
                .frame(minWidth: 1100, minHeight: 700)
        }
        .defaultSize(width: 1100, height: 850)
        .windowStyle(.titleBar)
    }
}
