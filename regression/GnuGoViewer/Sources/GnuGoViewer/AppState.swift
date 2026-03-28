import SwiftUI

@Observable
class AppState {
    var models: [RegressionModel] = []
    var currentTab: TabSelection = .wormsAndDragons
    var settings = MarkupSettings()
    var fullTestcaseText: String = ""
    var currentModelIndex: Int = 0

    var testcases: [String]
    var testcaseIndex: Int = 0
    var testcaseCommand: String = ""
    var completeTestcase: [String] = []

    init(engine: GtpEngine, testcases: [String]) {
        self.testcases = testcases
        let first = testcases.first ?? ""
        let model = RegressionModel(engine: engine, name: "Default engine")
        models.append(model)

        // Load first testcase in background
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            if self.excerptTestcase(first, engine: engine) {
                model.loadAndRunTestcase(lines: self.completeTestcase, command: self.testcaseCommand)
            }
            let cmd = self.testcaseCommand
            DispatchQueue.main.async {
                // Auto-select move generation tab when the test is a genmove command.
                if cmd.hasPrefix("reg_genmove") || cmd.hasPrefix("restricted_genmove") {
                    self.currentTab = .moveGeneration
                }
                self.refreshMarkup()
            }
        }
    }

    func excerptTestcase(_ spec: String, engine: GtpEngine) -> Bool {
        var filename: String
        var number: Int

        // Parse "filename:number"
        let parts = spec.components(separatedBy: ":")
        guard parts.count >= 2, let n = Int(parts.last ?? "") else { return false }
        number = n
        filename = parts.dropLast().joined(separator: ":")

        if !filename.hasSuffix(".tst") && !filename.hasSuffix(".sgf") {
            filename += ".tst"
        }

        // Resolve filename to an absolute path so that loadsgf SGF paths inside the .tst
        // file (which are relative to the regression directory) survive being sent to a
        // gnugo process that may have a different cwd (e.g. regression/GnuGoViewer/).
        let cwd = FileManager.default.currentDirectoryPath
        let absoluteFilename: String
        if filename.hasPrefix("/") {
            absoluteFilename = filename
        } else {
            absoluteFilename = URL(fileURLWithPath: cwd + "/" + filename).standardized.path
        }

        guard let contents = try? String(contentsOfFile: absoluteFilename, encoding: .utf8) else { return false }

        if filename.hasSuffix(".sgf") {
            let s = "loadsgf \(absoluteFilename) \(number)"
            let color = engine.sendCommand(s).text
            testcaseCommand = "reg_genmove \(color)"
            let text = "\(s)\n\(testcaseCommand)"
            completeTestcase = [s, testcaseCommand]
            if Thread.isMainThread {
                fullTestcaseText = text
                settings.expectedResult = ""
            } else {
                DispatchQueue.main.async {
                    self.fullTestcaseText = text
                    self.settings.expectedResult = ""
                }
            }
            return true
        }

        var fullLines: [String] = []
        var complete: [String] = []
        var found = false
        var parsedExpected = ""
        let lines = contents.components(separatedBy: "\n")

        for k in 0..<lines.count {
            let line = lines[k]
            guard !line.isEmpty else { continue }
            guard let firstChar = line.first else { continue }

            if firstChar >= "a" && firstChar <= "z" {
                if line.hasPrefix("loadsgf") {
                    // Resolve the SGF path to absolute so gnugo can find it regardless of cwd.
                    let tstDir = URL(fileURLWithPath: absoluteFilename).deletingLastPathComponent()
                    let parts = line.components(separatedBy: " ")
                    if parts.count >= 2 {
                        let sgfPath = parts[1]
                        let resolved = sgfPath.hasPrefix("/") ? sgfPath
                            : URL(fileURLWithPath: tstDir.path + "/" + sgfPath).standardized.path
                        complete = [(["loadsgf", resolved] + Array(parts.dropFirst(2))).joined(separator: " ")]
                    } else {
                        complete = [line]
                    }
                } else {
                    complete.append(line)
                }
            } else if let thisNum = Int(line.components(separatedBy: " ").first ?? ""),
                      thisNum == number {
                let cmd = line.components(separatedBy: " ").dropFirst().joined(separator: " ")
                testcaseCommand = cmd
                // Expected result from next line
                let nextLine = k + 1 < lines.count ? lines[k + 1] : ""
                if nextLine.hasPrefix("#? ["), let end = nextLine.firstIndex(of: "]") {
                    parsedExpected = String(nextLine[nextLine.index(nextLine.startIndex, offsetBy: 4)..<end])
                } else {
                    parsedExpected = ""
                }
                fullLines.append(lines[k])
                if k + 1 < lines.count { fullLines.append(lines[k + 1]) }
                found = true
                break
            }

            if "0123456789 ".contains(firstChar) {
                fullLines = []
            } else {
                fullLines.append(line)
            }
        }

        if found {
            completeTestcase = complete
            let text = fullLines.joined(separator: "\n")
            let er = parsedExpected
            if Thread.isMainThread {
                fullTestcaseText = text
                settings.expectedResult = er
            } else {
                DispatchQueue.main.async {
                    self.fullTestcaseText = text
                    self.settings.expectedResult = er
                }
            }
            return true
        }
        return false
    }

    func loadNewTestcase(_ spec: String) {
        guard !models.isEmpty else { return }
        let engine = models[0].engine
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            guard self.excerptTestcase(spec, engine: engine) else { return }
            for model in self.models {
                model.resetCaches()
                model.loadAndRunTestcase(lines: self.completeTestcase, command: self.testcaseCommand)
            }
            DispatchQueue.main.async { self.refreshMarkup() }
        }
    }

    func nextTestcase() {
        guard testcaseIndex < testcases.count - 1 else { return }
        testcaseIndex += 1
        loadNewTestcase(testcases[testcaseIndex])
    }

    func prevTestcase() {
        guard testcaseIndex > 0 else { return }
        testcaseIndex -= 1
        loadNewTestcase(testcases[testcaseIndex])
    }

    func selectNewEngine(path: String, name: String) {
        guard FileManager.default.fileExists(atPath: path) else { return }
        let cmd = [path, "--quiet", "--mode", "gtp", "-w", "-t", "-d0x101840"]
        let newEngine = GtpEngine(command: cmd)
        let newModel = RegressionModel(engine: newEngine, name: name)
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            newModel.loadAndRunTestcase(lines: self.completeTestcase, command: self.testcaseCommand)
            DispatchQueue.main.async {
                self.models.append(newModel)
                self.refreshMarkup()
            }
        }
    }

    func boardTapped(vertex: String, modelIndex: Int) {
        guard !vertex.isEmpty, modelIndex < models.count else { return }
        let model = models[modelIndex]
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            switch self.currentTab {
            case .wormsAndDragons:
                switch self.settings.wormDragonDataMode {
                case .wormData:    model.showWormData(vertex: vertex)
                case .dragonData1: model.showDragonData(vertex: vertex, part: 1)
                case .dragonData2: model.showDragonData(vertex: vertex, part: 2)
                }
            case .moveGeneration:
                if self.settings.moveGenMode == .deltaTerritory {
                    DispatchQueue.main.async {
                        self.settings.deltaVertex = vertex
                        self.refreshMarkup()
                    }
                } else {
                    model.showMoveReasons(vertex: vertex)
                }
            case .eyes:
                let color = self.settings.eyeColor == .white ? "white" : "black"
                model.showEyeData(vertex: vertex, color: color)
            case .influence:
                if self.settings.influenceSource == .afterMove
                    || self.settings.influenceSource == .followup {
                    DispatchQueue.main.async {
                        self.settings.moveInfluenceVertex = vertex
                        self.refreshMarkup()
                    }
                }
            case .reading:
                self.handleReadingTap(vertex: vertex, model: model)
            }
        }
    }

    private func handleReadingTap(vertex: String, model: RegressionModel) {
        let mode = settings.readingMode
        let needsTwo = mode.needsTwoVertices

        // Clear markup when no first vertex selected or when using single-vertex modes.
        let shouldClear = (settings.firstVertex == nil || settings.firstVertex!.isEmpty)
            || mode == .tactical || mode == .owl
        DispatchQueue.main.async {
            if shouldClear { model.goban.clearMarkup() }
            model.goban.addSymbol(vertex: vertex, symbol: .bigDot, color: .green)
            model.goban.commitMarkup()   // already on main thread — applies immediately
        }

        // Mirror Pike line 1210-1211: append ".engine_name" to SGF filename when
        // multiple engines are running simultaneously.
        func sgfFileForModel(_ base: String) -> String {
            guard !base.isEmpty, models.count > 1 else { return base }
            return base + "." + model.name.replacing(" ", with: "_")
        }

        if mode == .tactical || mode == .owl {
            let prefix = mode == .owl ? "owl_" : ""
            let reset = mode == .owl ? "reset_owl_node_counter" : "reset_reading_node_counter"
            let get   = mode == .owl ? "get_owl_node_counter"   : "get_reading_node_counter"
            let sgfFile = settings.saveSgf ? sgfFileForModel(settings.sgfFile) : ""
            let viewer  = settings.openSgfViewer ? settings.sgfViewerCmd : ""
            model.doReading(resetCounter: reset, getCounter: get,
                            sgfFile: sgfFile, sgfViewerCmd: viewer,
                            firstCommand: "\(prefix)attack \(vertex)",
                            secondCommand: "\(prefix)defend \(vertex)")
        } else if needsTwo {
            if settings.firstVertex == nil || settings.firstVertex!.isEmpty {
                DispatchQueue.main.async { self.settings.firstVertex = vertex }
            } else if settings.firstVertex != vertex {
                let fv = settings.firstVertex!
                let sgfFile = settings.saveSgf ? sgfFileForModel(settings.sgfFile) : ""
                let viewer  = settings.openSgfViewer ? settings.sgfViewerCmd : ""
                let (c1, c2, reset, get): (String, String, String, String) = {
                    switch mode {
                    case .owlDoesAttack:
                        return ("owl_does_attack \(fv) \(vertex)", "",
                                "reset_owl_node_counter", "get_owl_node_counter")
                    case .owlDoesDefend:
                        return ("owl_does_defend \(fv) \(vertex)", "",
                                "reset_owl_node_counter", "get_owl_node_counter")
                    case .connection:
                        return ("connect \(fv) \(vertex)", "disconnect \(fv) \(vertex)",
                                "reset_connection_node_counter", "get_connection_node_counter")
                    default: // semeai
                        return ("analyze_semeai \(fv) \(vertex)", "analyze_semeai \(vertex) \(fv)",
                                "reset_owl_node_counter", "get_owl_node_counter")
                    }
                }()
                model.doReading(resetCounter: reset, getCounter: get,
                                sgfFile: sgfFile, sgfViewerCmd: viewer,
                                firstCommand: c1, secondCommand: c2)
                DispatchQueue.main.async { self.settings.firstVertex = nil }
            }
        }
    }

    func refreshMarkup() {
        let s = settings
        let idx = currentModelIndex
        DispatchQueue.global().async { [weak self] in
            guard let self = self, idx < self.models.count else { return }
            self.models[idx].addMarkup(tab: self.currentTab, settings: s)
        }
    }
}
