// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "GnuGoViewer",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "GnuGoViewer",
            path: "Sources/GnuGoViewer"
        )
    ]
)
