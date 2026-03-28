import Foundation
import SwiftUI

@Observable
class RegressionModel {
    let engine: GtpEngine
    let goban: GobanModel
    let name: String

    // Observable properties — must only be mutated on main thread.
    var dataText: String = ""
    var dataTitle: String = ""
    var result: String = ""

    // traces is appended from the stderr thread and reset from the GTP queue;
    // protect both with tracesLock.
    private let tracesLock = NSLock()
    var traces: [String] = []
    // Non-@Published mirror of result; safe to read from any thread after handleTestcase returns.
    private(set) var resultSnapshot: String = ""
    private var worms: [String: [String]] = [:]
    private var dragons: [String: [String]] = [:]
    private var wormsInitialized = false
    private var dragonsInitialized = false
    private var wormAndDragonCache: [String: String] = [:]

    private(set) var eyeData: [String: [String: String]]?
    var halfEyeData: [String: String] = [:]
    private(set) var eyeTypes: [String: [String: String]]?

    var completeTest: [String] = []
    var testcaseCommand: String = ""
    var colorToMove: String = "black"

    init(engine: GtpEngine, name: String) {
        self.engine = engine
        self.name = name
        self.goban = GobanModel(boardSize: 19)
    }

    // MARK: - Lifecycle

    func resetCaches() {
        wormsInitialized = false
        dragonsInitialized = false
        worms = [:]
        dragons = [:]
        wormAndDragonCache = [:]
        eyeData = nil
        eyeTypes = nil
    }

    /// Load testcase lines, refresh the board, and run the test command.
    func loadAndRunTestcase(lines: [String], command: String) {
        completeTest = lines
        testcaseCommand = command
        loadTestcase(lines: lines)
        refreshBoard()
        handleTestcase(command: command)
    }

    @discardableResult
    func send(_ cmd: String) -> String {
        engine.sendCommand(cmd).text
    }

    // MARK: - Test loading

    private func loadTestcase(lines: [String]) {
        for line in lines {
            let first = line.first.map(String.init) ?? ""
            // Skip comment lines, blank lines, and numbered test lines
            if !"0123456789 #".contains(first) && !first.isEmpty {
                let response = send(line)
                if line.hasPrefix("loadsgf") {
                    colorToMove = response.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
        }
    }

    /// Sends the test command, collects traces. Sets self.result on main thread.
    @discardableResult
    func handleTestcase(command: String) -> String {
        tracesLock.lock()
        traces = []
        tracesLock.unlock()
        engine.traceCallback = { [weak self] s in
            guard let self = self else { return }
            self.tracesLock.lock()
            self.traces.append(s)
            self.tracesLock.unlock()
        }
        let r = send(command)
        engine.traceCallback = nil
        resultSnapshot = r
        DispatchQueue.main.async { self.result = r }
        return r
    }

    /// Query board size and stones; update GobanModel on main thread.
    func refreshBoard() {
        let sizeStr = send("query_boardsize")
        let size = Int(sizeStr.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 19
        let whites = send("list_stones white")
            .components(separatedBy: " ")
            .filter { !$0.isEmpty && $0.uppercased() != "PASS" }
            .map { $0.uppercased() }
        let blacks = send("list_stones black")
            .components(separatedBy: " ")
            .filter { !$0.isEmpty && $0.uppercased() != "PASS" }
            .map { $0.uppercased() }

        goban.stageBoardSize(size)
        goban.stageStones(white: Set(whites), black: Set(blacks))
    }

    // MARK: - Worms & Dragons

    func getWorms() {
        worms = [:]
        for line in send("worm_stones").components(separatedBy: "\n") {
            let parts = line.components(separatedBy: " ").filter { !$0.isEmpty }
            if let first = parts.first { worms[first] = parts }
        }
        wormsInitialized = true
    }

    func getDragons() {
        dragons = [:]
        for line in send("dragon_stones").components(separatedBy: "\n") {
            let parts = line.components(separatedBy: " ").filter { !$0.isEmpty }
            if let first = parts.first { dragons[first] = parts }
        }
        dragonsInitialized = true
    }

    func getWormOrDragonData(type: String, field: String, vertex: String) -> String {
        let cmd = "\(type)_data \(vertex)"
        let data: String
        if let cached = wormAndDragonCache[cmd] {
            data = cached
        } else {
            data = send(cmd)
            wormAndDragonCache[cmd] = data
        }
        for row in data.components(separatedBy: "\n") {
            if row.hasPrefix(field) {
                let parts = row.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
                return parts.last ?? ""
            }
        }
        return ""
    }

    // MARK: - Data display helpers (publish to main thread)

    func showWormData(vertex: String) {
        let raw = send("worm_data \(vertex)")
        let text = raw.components(separatedBy: "\n")
            .dropFirst()
            .map { formatDataLine($0) }
            .joined(separator: "\n")
        let title = "Worm data for \(vertex)"
        DispatchQueue.main.async { self.dataText = text; self.dataTitle = title }
    }

    func showDragonData(vertex: String, part: Int) {
        let raw = send("dragon_data \(vertex)")
        let allLines = raw.components(separatedBy: "\n")
        let slice = part == 1 ? Array(allLines.dropFirst().prefix(20))
                              : Array(allLines.dropFirst(21))
        let text = slice.map { formatDataLine($0) }.joined(separator: "\n")
        let title = "Dragon data for \(vertex)"
        DispatchQueue.main.async { self.dataText = text; self.dataTitle = title }
    }

    func showMoveReasons(vertex: String) {
        let raw = send("move_reasons \(vertex)")
        var lines = raw.components(separatedBy: "\n")

        // Snapshot traces under lock to avoid race with traceCallback.
        tracesLock.lock()
        let traces = self.traces
        tracesLock.unlock()

        // Find move generation trace lines
        var k = traces.count - 1
        while k >= 0 {
            if traces[k].contains("Move generation values \(vertex) to ") { break }
            k -= 1
        }
        if k >= 0 {
            var interesting = [traces[k]]
            k -= 1
            while k >= 0 {
                if traces[k].hasPrefix("  \(vertex): ") {
                    interesting.append(traces[k])
                } else if !traces[k].hasPrefix("    \(vertex): ") {
                    break
                }
                k -= 1
            }
            lines += [""] + interesting.reversed()
        }

        // Pattern matches
        var firstPattern = true
        var addContinuation = false
        for line in traces {
            if line.contains("pattern ") && line.contains(" matched at \(vertex)") {
                if firstPattern { lines += [""]; firstPattern = false }
                addContinuation = true
                lines.append(line)
            } else if line.hasPrefix("...") && addContinuation {
                lines.append(line)
            } else {
                addContinuation = false
            }
        }

        // Blunder devaluation
        for line in traces {
            if line.hasPrefix("Move at \(vertex) is") { lines.append(line) }
        }

        let text = lines.joined(separator: "\n")
        let title = "Move reasons for \(vertex)"
        DispatchQueue.main.async { self.dataText = text; self.dataTitle = title }
    }

    func showEyeData(vertex: String, color: String) {
        var lines: [String] = []
        if let colorData = eyeData?[color], let raw = colorData[vertex] {
            lines = raw.components(separatedBy: "\n").map { formatDataLine($0) }
            if let half = halfEyeData[vertex] {
                lines += [""] + half.components(separatedBy: "\n").map { formatDataLine($0) }
            }
        }
        let text = lines.joined(separator: "\n")
        let title = "\(color) eye data for \(vertex)"
        DispatchQueue.main.async { self.dataText = text; self.dataTitle = title }
    }

    func doReading(resetCounter: String, getCounter: String,
                   sgfFile: String, sgfViewerCmd: String,
                   firstCommand: String, secondCommand: String) {
        send("clear_cache")
        let useFile = !sgfFile.isEmpty
        if useFile { send("start_sgftrace") }
        send(resetCounter)
        let r1 = send(firstCommand)
        let nodes1 = send(getCounter)
        var lines = ["\(firstCommand)\t\(r1)\t(\(nodes1) nodes)"]

        if !r1.hasPrefix("0") && !secondCommand.isEmpty {
            send(resetCounter)
            let r2 = send(secondCommand)
            let nodes2 = send(getCounter)
            lines.append("\(secondCommand)\t\(r2)\t(\(nodes2) nodes)")
        }

        if useFile {
            send("finish_sgftrace \(sgfFile)")
            if !sgfViewerCmd.isEmpty {
                let parts = sgfViewerCmd.replacing("%s", with: sgfFile)
                    .components(separatedBy: " ")
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: parts[0])
                proc.arguments = Array(parts.dropFirst())
                try? proc.run()
            }
        }
        let text = lines.joined(separator: "\n")
        DispatchQueue.main.async { self.dataText = text; self.dataTitle = "Reading result" }
    }

    // MARK: - Markup computation
    // All functions manipulate the goban staging buffer (background-thread safe).
    // addMarkup() commits the staging buffer to the main thread at the end.

    func addMarkup(tab: TabSelection, settings: MarkupSettings) {
        goban.clearMarkup()
        switch tab {
        case .wormsAndDragons: addWormsAndDragonsMarkup(settings: settings)
        case .moveGeneration:  addMoveGenerationMarkup(settings: settings)
        case .eyes:            addEyesMarkup(settings: settings)
        case .influence:       addInfluenceMarkup(settings: settings)
        case .reading:         addReadingMarkup(settings: settings)
        }
        goban.commitMarkup()   // single dispatch to main thread
    }

    func addWormsAndDragonsMarkup(settings: MarkupSettings) {
        let statusColors: [String: Color] = [
            "alive": .green, "critical": .yellow, "dead": .red, "unknown": .blue
        ]
        let safetyColors: [String: Color] = [
            "alive": .green, "critical": .yellow, "dead": .red,
            "tactically dead": colorFromString("brown"),
            "alive in seki": .cyan, "strongly alive": .blue,
            "invincible": .purple, "inessential": .orange
        ]

        switch settings.wormDragonOverlay {
        case .dragonStatus:
            if !dragonsInitialized { getDragons() }
            for (dragon, stones) in dragons {
                let status = getWormOrDragonData(type: "dragon", field: "status", vertex: dragon)
                let color = statusColors[status] ?? .blue
                for stone in stones { goban.addSymbol(vertex: stone, symbol: .dot, color: color) }
            }
        case .dragonSafety:
            if !dragonsInitialized { getDragons() }
            for (dragon, stones) in dragons {
                let safety = getWormOrDragonData(type: "dragon", field: "safety", vertex: dragon)
                let color = safetyColors[safety] ?? .blue
                for stone in stones { goban.addSymbol(vertex: stone, symbol: .dot, color: color) }
            }
        case .wormStatus:
            if !wormsInitialized { getWorms() }
            for (worm, stones) in worms {
                let attack = getWormOrDragonData(type: "worm", field: "attack_code", vertex: worm)
                let defense = getWormOrDragonData(type: "worm", field: "defense_code", vertex: worm)
                let status: String
                if attack != "0" {
                    status = defense == "0" ? "dead" : "critical"
                } else {
                    status = "alive"
                }
                let color = statusColors[status] ?? .blue
                for stone in stones { goban.addSymbol(vertex: stone, symbol: .dot, color: color) }
            }
        case .none:
            break
        }
    }

    func addMoveGenerationMarkup(settings: MarkupSettings) {
        let isGenMove = testcaseCommand.hasPrefix("reg_genmove")
            || testcaseCommand.hasPrefix("restricted_genmove")

        if (settings.moveGenMode == .topMoves || settings.moveGenMode == .allMoves) && isGenMove {
            var answers = settings.expectedResult
            var color: Color = .green
            if answers.hasPrefix("!") { answers = String(answers.dropFirst()); color = .red }

            if testcaseCommand.hasPrefix("restricted_genmove") {
                let allowed = testcaseCommand.components(separatedBy: " ").dropFirst(2)
                for move in allowed {
                    let isInAnswers = answers.components(separatedBy: "|").contains(move)
                    let c: Color = (color == .green) == isInAnswers ? .green : .red
                    goban.addSymbol(vertex: move, symbol: .triangle, color: c)
                }
            } else {
                for answer in answers.components(separatedBy: "|").filter({ !$0.isEmpty }) {
                    goban.addSymbol(vertex: answer, symbol: .bigSquare, color: color)
                }
            }

            let parts = testcaseCommand.components(separatedBy: " ")
            let moveColor = parts.count > 1 ? parts[1].lowercased() : "black"
            // Use resultSnapshot (non-@Published) instead of result to avoid off-main read.
            goban.addSymbol(vertex: resultSnapshot, symbol: .stone,
                            color: moveColor == "white" ? .white : .black)
        }

        // For non-genmove test cases (e.g. defend, attack), genmove() was never
        // called so best_moves[] is empty.  Trigger move generation so that
        // top_moves / all_move_values have data to return.
        // Re-enable trace capture so delta territory can read the output.
        if !isGenMove {
            tracesLock.lock()
            traces = []
            tracesLock.unlock()
            engine.traceCallback = { [weak self] s in
                guard let self = self else { return }
                self.tracesLock.lock()
                self.traces.append(s)
                self.tracesLock.unlock()
            }
            send("reg_genmove \(colorToMove)")
            engine.traceCallback = nil
        }

        // Snapshot traces under lock before iterating.
        tracesLock.lock()
        let traces = self.traces
        tracesLock.unlock()

        switch settings.moveGenMode {
        case .topMoves:
            let parts = send("top_moves").components(separatedBy: " ")
            var i = 0
            while i + 1 < parts.count {
                goban.addText(vertex: parts[i], text: parts[i + 1], color: .blue)
                i += 2
            }
        case .allMoves:
            for line in send("all_move_values").components(separatedBy: "\n") {
                let p = line.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
                if p.count >= 2 { goban.addText(vertex: p[0], text: p[1], color: .blue) }
            }
        case .deltaTerritory:
            let move = settings.deltaVertex
            guard move != "PASS" && !move.isEmpty else { return }
            goban.addSymbol(vertex: move, symbol: .stone, color: .gray)
            var textLines: [String] = []
            var k = traces.count - 1
            while k >= 0 {
                if traces[k].contains("\(move): ") && traces[k].contains("change in territory")
                    && !traces[k].contains("cached") { break }
                k -= 1
            }
            if k >= 0 {
                textLines.append(traces[k])
                k -= 1
                while k >= 0 {
                    guard traces[k].hasPrefix("    ") else { break }
                    textLines.insert(traces[k], at: 0)
                    // Trace format: "    G1:   - H1 territory change 1.00 (-1.00 -> -0.00)"
                    //   p: [G1:, -, H1, territory, change, 1.00, ...]
                    // Or:  "    G1:   - captured stones 1.00"
                    //   p: [G1:, -, captured, stones, 1.00]
                    let p = traces[k].components(separatedBy: .whitespaces).filter { !$0.isEmpty }
                    if p.count >= 6, p[1] == "-", p[3] == "territory", p[4] == "change",
                       let val = Double(p[5]) {
                        let vx = p[2]
                        let label = String(format: "%.1f", val)
                        goban.addText(vertex: vx, text: label, color: val < 0 ? .red : .blue)
                    }
                    k -= 1
                }
            }
            let text = textLines.joined(separator: "\n")
            let title = "Delta territory for \(move)"
            DispatchQueue.main.async { self.dataText = text; self.dataTitle = title }
        case .none:
            break
        }
    }

    func addEyesMarkup(settings: MarkupSettings) {
        if eyeData == nil { computeEyeData() }
        let color = settings.eyeColor == .white ? "white" : "black"
        let properColor: Color = color == "white" ? .white : .black
        let grayish = color == "white" ? Color(red: 0.75, green: 0.75, blue: 0.75)
                                       : Color(red: 0.25, green: 0.25, blue: 0.25)
        guard let types = eyeTypes?[color] else { return }
        for (vertex, eyeType) in types {
            switch eyeType {
            case "proper":   goban.addSymbol(vertex: vertex, symbol: .bigDot,  color: properColor)
            case "half":     goban.addSymbol(vertex: vertex, symbol: .bigDot,  color: grayish)
            case "marginal": goban.addSymbol(vertex: vertex, symbol: .dot,     color: grayish)
            default: break
            }
        }
    }

    private func computeEyeData() {
        var ed: [String: [String: String]] = ["white": [:], "black": [:]]
        var et: [String: [String: String]] = ["white": [:], "black": [:]]
        halfEyeData = [:]
        // Use a snapshot of boardSize to avoid reading @Published from background.
        // GobanModel.boardSize is only written from main via stageBoardSize, and
        // this function runs on a background queue after refreshBoard() completes,
        // so the value is stable during the loop.
        let n = goban.boardSize
        for y in 0..<n {
            for x in 0..<n {
                let vertex = goban.xyToVertex(x: x, y: y)
                var isMarginal: Set<String> = []
                for color in ["white", "black"] {
                    let raw = send("eye_data \(color) \(vertex)")
                    // Skip vertices whose eye origin is PASS (not an eye point)
                    guard !raw.contains("PASS") else { continue }
                    ed[color, default: [:]][vertex] = raw
                    // Check for "marginal  1" (value after "marginal" key)
                    for row in raw.components(separatedBy: "\n") {
                        if row.hasPrefix("marginal") {
                            let val = row.components(separatedBy: .whitespaces)
                                .filter { !$0.isEmpty }.last ?? "0"
                            if val == "1" { isMarginal.insert(color) }
                        }
                    }
                }
                guard ed["white"]?[vertex] != nil || ed["black"]?[vertex] != nil
                else { continue }

                let halfRaw = send("half_eye_data \(vertex)")
                // Store if not type 0 (i.e., it IS a half-eye)
                var isHalfEye = false
                for row in halfRaw.components(separatedBy: "\n") {
                    if row.hasPrefix("type") {
                        let val = row.components(separatedBy: .whitespaces)
                            .filter { !$0.isEmpty }.last ?? "0"
                        if val != "0" { isHalfEye = true }
                    }
                }
                if isHalfEye { halfEyeData[vertex] = halfRaw }

                for color in ["white", "black"] {
                    guard ed[color]?[vertex] != nil else { continue }
                    if isMarginal.contains(color) {
                        et[color, default: [:]][vertex] = "marginal"
                    } else if isHalfEye {
                        et[color, default: [:]][vertex] = "half"
                    } else {
                        et[color, default: [:]][vertex] = "proper"
                    }
                }
            }
        }
        eyeData = ed
        eyeTypes = et
    }

    func addInfluenceMarkup(settings: MarkupSettings) {
        let command: String
        switch settings.influenceSource {
        case .whiteKnown:  command = "initial_influence white"
        case .blackKnown:  command = "initial_influence black"
        case .afterMove:
            let move = settings.moveInfluenceVertex
            guard move != "PASS" && !move.isEmpty else { return }
            goban.addSymbol(vertex: move, symbol: .stone, color: .gray)
            command = "move_influence \(settings.moveColor) \(move)"
        case .followup:
            let move = settings.moveInfluenceVertex
            guard move != "PASS" && !move.isEmpty else { return }
            goban.addSymbol(vertex: move, symbol: .stone, color: .gray)
            command = "followup_influence \(settings.moveColor) \(move)"
        }

        // move_influence and followup_influence call prepare_move_influence_debugging()
        // which requires genmove() to have populated move reasons for the SAME color.
        if settings.influenceSource == .afterMove || settings.influenceSource == .followup {
            send("reg_genmove \(settings.moveColor)")
        }

        let whatData = settings.influenceData.gtpCommand

        let raw = send("\(command) \(whatData)")
        let values = raw.replacing("\n", with: " ")
            .components(separatedBy: " ").filter { !$0.isEmpty }
        let n = goban.boardSize
        var k = 0
        for y in 0..<n {
            for x in 0..<n {
                guard k < values.count else { break }
                let vertex = goban.xyToVertex(x: x, y: y)
                let valStr = values[k]; k += 1

                if whatData == "influence_regions" {
                    let val = Int(valStr) ?? 0
                    let (sym, col): (MarkupSymbol, Color) = {
                        switch val {
                        case 3:  return (.bigDot, .white)
                        case 2:  return (.bigDot, Color(red: 0.75, green: 0.75, blue: 0.75))
                        case 1:  return (.dot,    Color(red: 0.75, green: 0.75, blue: 0.75))
                        case -1: return (.dot,    Color(red: 0.25, green: 0.25, blue: 0.25))
                        case -2: return (.bigDot, Color(red: 0.25, green: 0.25, blue: 0.25))
                        case -3: return (.bigDot, .black)
                        default: return (.dot, .clear)
                        }
                    }()
                    if val != 0 { goban.addSymbol(vertex: vertex, symbol: sym, color: col) }
                } else if whatData.contains("permeability") {
                    if let v = Double(valStr), v != 1.0 {
                        goban.addText(vertex: vertex, text: valStr, color: .blue)
                    }
                } else if whatData == "non_territory" {
                    if !goban.isOccupied(vertex) {
                        switch Int(valStr) ?? 0 {
                        case 1: goban.addSymbol(vertex: vertex, symbol: .dot, color: .black)
                        case 2: goban.addSymbol(vertex: vertex, symbol: .dot, color: .white)
                        case 0: goban.addSymbol(vertex: vertex, symbol: .dot, color: .gray)
                        default: break
                        }
                    }
                } else if let v = Double(valStr) {
                    if v > 0 { goban.addText(vertex: vertex, text: valStr, color: .blue) }
                    else if v < 0 { goban.addText(vertex: vertex, text: valStr, color: .red) }
                }
            }
        }
    }

    func addReadingMarkup(settings: MarkupSettings) {
        if settings.readingMode.needsTwoVertices, let fv = settings.firstVertex, !fv.isEmpty {
            goban.addSymbol(vertex: fv, symbol: .bigDot, color: .green)
        }
    }

    // MARK: - Private helpers

    private func formatDataLine(_ s: String) -> String {
        let parts = s.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        guard parts.count >= 2 else { return s }
        return String(format: "%-20@ %@",
                      parts[0] as NSString,
                      parts[1...].joined(separator: " ") as NSString)
    }
}
