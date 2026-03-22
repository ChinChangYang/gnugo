import Foundation
import SwiftUI

// Go board letters: A-Z minus I
let goLetters: [Character] = Array("ABCDEFGHJKLMNOPQRSTUVWXYZ")

enum MarkupSymbol: Equatable {
    case circle
    case square
    case bigSquare
    case triangle
    case dot
    case bigDot
    case smallDot
    case stone
    case text(String)
}

struct Markup {
    var vertex: String
    var symbol: MarkupSymbol
    var color: Color
}

@Observable
class GobanModel {
    // Observable properties — must only be written on main thread.
    var boardSize: Int
    var whiteStones: Set<String> = []
    var blackStones: Set<String> = []
    var markups: [Markup] = []

    // Staging buffers — written on any thread, committed to @Published in one
    // DispatchQueue.main.async call to avoid per-mutation main-thread hops.
    private let markupLock = NSLock()
    private var stagingMarkups: [Markup] = []

    init(boardSize: Int) {
        self.boardSize = boardSize
    }

    // MARK: - Stone management (call commitStones on main thread after)

    func stageStones(white: Set<String>, black: Set<String>) {
        // Caller is responsible for dispatching to main before reading Published.
        let w = white
        let b = black
        DispatchQueue.main.async {
            self.whiteStones = w
            self.blackStones = b
        }
    }

    func stageBoardSize(_ size: Int) {
        DispatchQueue.main.async { self.boardSize = size }
    }

    // MARK: - Markup staging (background-thread safe)

    /// Clear the staging buffer (background thread OK).
    func clearMarkup() {
        markupLock.lock()
        stagingMarkups = []
        markupLock.unlock()
    }

    func addSymbol(vertex: String, symbol: MarkupSymbol, color: Color) {
        let v = vertex.uppercased()
        if v == "PASS" || v.isEmpty { return }
        markupLock.lock()
        stagingMarkups.append(Markup(vertex: v, symbol: symbol, color: color))
        markupLock.unlock()
    }

    func addText(vertex: String, text: String, color: Color) {
        addSymbol(vertex: vertex, symbol: .text(text), color: color)
    }

    /// Publish the staging buffer to the UI. Safe to call from any thread:
    /// applies immediately if already on main, dispatches async otherwise.
    func commitMarkup() {
        markupLock.lock()
        let batch = stagingMarkups
        markupLock.unlock()
        if Thread.isMainThread {
            markups = batch
        } else {
            DispatchQueue.main.async { self.markups = batch }
        }
    }

    // MARK: - Read-only accessors (safe from any thread)

    func isOccupied(_ vertex: String) -> Bool {
        let v = vertex.uppercased()
        return whiteStones.contains(v) || blackStones.contains(v)
    }

    /// Convert GTP vertex (e.g. "A1", "Q16") to (col, row) zero-indexed.
    /// col 0 = leftmost (A), row 0 = top (boardSize row).
    func vertexToXY(_ vertex: String) -> (Int, Int)? {
        let v = vertex.uppercased()
        guard !v.isEmpty, let letterIdx = goLetters.firstIndex(of: v.first!) else { return nil }
        let col = letterIdx
        guard let row = Int(v.dropFirst()) else { return nil }
        let y = boardSize - row
        guard col >= 0, col < boardSize, y >= 0, y < boardSize else { return nil }
        return (col, y)
    }

    func xyToVertex(x: Int, y: Int) -> String {
        guard x >= 0, x < goLetters.count, y >= 0, y < boardSize else { return "" }
        return "\(goLetters[x])\(boardSize - y)"
    }

    // MARK: - Pixel coordinate math (mirrors Pike Goban)

    func spacing(boardPixelSize: CGFloat) -> CGFloat {
        boardPixelSize / (CGFloat(boardSize) + 1.5)
    }

    func offset(boardPixelSize: CGFloat) -> CGFloat {
        let sp = spacing(boardPixelSize: boardPixelSize)
        return (boardPixelSize - sp * CGFloat(boardSize - 1)) / 2
    }

    func pixelToVertex(px: CGFloat, py: CGFloat, boardPixelSize: CGFloat) -> String {
        let sp = spacing(boardPixelSize: boardPixelSize)
        let off = offset(boardPixelSize: boardPixelSize)
        let x = Int(floor((px - off + sp / 2) / sp))
        let y = Int(floor((py - off + sp / 2) / sp))
        guard x >= 0, x < boardSize, y >= 0, y < boardSize else { return "" }
        return xyToVertex(x: x, y: y)
    }

    func vertexToPixel(vertex: String, boardPixelSize: CGFloat) -> CGPoint? {
        guard let (x, y) = vertexToXY(vertex) else { return nil }
        let sp = spacing(boardPixelSize: boardPixelSize)
        let off = offset(boardPixelSize: boardPixelSize)
        return CGPoint(x: off + CGFloat(x) * sp, y: off + CGFloat(y) * sp)
    }

    func hoshiPoints() -> [(Int, Int)] {
        let a = 2 + (boardSize >= 12 ? 1 : 0)
        let b = boardSize - a - 1
        let c = boardSize / 2
        var pts: [(Int, Int)] = []

        if (boardSize % 2 == 0 && boardSize >= 8) || (boardSize % 2 == 1 && boardSize >= 9) {
            pts += [(a, a), (a, b), (b, a), (b, b)]
        }
        if boardSize % 2 == 1 && boardSize >= 5 {
            pts.append((c, c))
        }
        if boardSize % 2 == 1 && boardSize >= 13 {
            pts += [(a, c), (b, c), (c, a), (c, b)]
        }
        return pts
    }
}

// MARK: - Color helper
func colorFromString(_ name: String) -> Color {
    let lower = name.lowercased()
    switch lower {
    case "green":   return .green
    case "red":     return .red
    case "blue":    return .blue
    case "white":   return .white
    case "black":   return .black
    case "gray", "grey": return .gray
    case "yellow":  return .yellow
    case "brown":   return Color(red: 0.6, green: 0.3, blue: 0.1)
    case "cyan":    return .cyan
    case "purple":  return .purple
    case "orange":  return .orange
    default:
        // Try hex like #c0c0c0
        if name.hasPrefix("#") {
            let hex = String(name.dropFirst())
            if hex.count == 6, let value = UInt64(hex, radix: 16) {
                let r = Double((value >> 16) & 0xFF) / 255
                let g = Double((value >> 8) & 0xFF) / 255
                let b = Double(value & 0xFF) / 255
                return Color(red: r, green: g, blue: b)
            }
        }
        return .gray
    }
}
