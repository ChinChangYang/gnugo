import SwiftUI

// MARK: - Settings enums

enum WormDragonDataMode: Int, CaseIterable {
    case wormData, dragonData1, dragonData2
    var label: String {
        switch self {
        case .wormData:    return "worm data"
        case .dragonData1: return "dragon data, part 1"
        case .dragonData2: return "dragon data, part 2"
        }
    }
}

enum WormDragonOverlay {
    case none, wormStatus, dragonStatus, dragonSafety
}

enum MoveGenMode {
    case none, topMoves, allMoves, deltaTerritory
}

enum EyeColor {
    case white, black
}

enum InfluenceSource {
    case whiteKnown, blackKnown, afterMove, followup
}

enum InfluenceData: CaseIterable {
    case regions, territoryValue, whiteInfluence, blackInfluence
    case whiteStrength, blackStrength, whitePermeability, blackPermeability
    case whiteAttenuation, blackAttenuation, nonTerritory

    var label: String {
        switch self {
        case .regions:          return "influence regions"
        case .territoryValue:   return "territory value"
        case .whiteInfluence:   return "white influence"
        case .blackInfluence:   return "black influence"
        case .whiteStrength:    return "white strength"
        case .blackStrength:    return "black strength"
        case .whitePermeability: return "white permeability"
        case .blackPermeability: return "black permeability"
        case .whiteAttenuation: return "white attenuation"
        case .blackAttenuation: return "black attenuation"
        case .nonTerritory:     return "non-territory"
        }
    }
}

enum ReadingMode {
    case tactical, owl, owlDoesAttack, owlDoesDefend, connection, semeai
}

struct MarkupSettings {
    // Worms & Dragons tab
    var wormDragonDataMode: WormDragonDataMode = .wormData
    var wormDragonOverlay: WormDragonOverlay = .none
    // Move generation tab
    var moveGenMode: MoveGenMode = .none
    var deltaVertex: String = "PASS"
    var expectedResult: String = ""
    // Eyes tab
    var eyeColor: EyeColor = .white
    // Influence tab
    var influenceSource: InfluenceSource = .whiteKnown
    var influenceData: InfluenceData = .regions
    var moveInfluenceVertex: String = "PASS"
    var moveColor: String = "black"
    // Reading tab
    var readingMode: ReadingMode = .tactical
    var saveSgf: Bool = true
    var sgfFile: String = "vars.sgf"
    var openSgfViewer: Bool = false
    var sgfViewerCmd: String = "open %s"
    var firstVertex: String? = nil
}

// MARK: - AppState

class AppState: ObservableObject {
    @Published var models: [RegressionModel] = []
    @Published var currentTabIndex: Int = 0
    @Published var settings = MarkupSettings()
    @Published var fullTestcaseText: String = ""
    @Published var currentModelIndex: Int = 0

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
                model.completeTest = self.completeTestcase
                model.testcaseCommand = self.testcaseCommand
                model.loadTestcase(lines: self.completeTestcase)
                model.refreshBoard()
                model.handleTestcase(command: self.testcaseCommand)
            }
            let cmd = self.testcaseCommand
            DispatchQueue.main.async {
                // Auto-select move generation tab when the test is a genmove command.
                if cmd.hasPrefix("reg_genmove") || cmd.hasPrefix("restricted_genmove") {
                    self.currentTabIndex = 1
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
        guard parts.count >= 2, let n = Int(parts.last!) else { return false }
        number = n
        filename = parts.dropLast().joined(separator: ":")

        if !filename.hasSuffix(".tst") && !filename.hasSuffix(".sgf") {
            filename += ".tst"
        }

        guard let contents = try? String(contentsOfFile: filename, encoding: .utf8) else { return false }

        if filename.hasSuffix(".sgf") {
            let s = "loadsgf \(filename) \(number)"
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
            let firstChar = line.first!

            if firstChar >= "a" && firstChar <= "z" {
                if line.hasPrefix("loadsgf") {
                    complete = [line]
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
        guard excerptTestcase(spec, engine: engine) else { return }

        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            for model in self.models {
                model.wormsInitialized = false
                model.dragonsInitialized = false
                model.worms = [:]
                model.dragons = [:]
                model.wormAndDragonCache = [:]
                model.eyeData = nil
                model.eyeTypes = nil
                model.completeTest = self.completeTestcase
                model.testcaseCommand = self.testcaseCommand
                model.loadTestcase(lines: self.completeTestcase)
                model.refreshBoard()
                model.handleTestcase(command: self.testcaseCommand)
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
            newModel.completeTest = self.completeTestcase
            newModel.testcaseCommand = self.testcaseCommand
            newModel.loadTestcase(lines: self.completeTestcase)
            newModel.refreshBoard()
            newModel.handleTestcase(command: self.testcaseCommand)
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
            switch self.currentTabIndex {
            case 0: // Worms & dragons
                switch self.settings.wormDragonDataMode {
                case .wormData:    model.showWormData(vertex: vertex)
                case .dragonData1: model.showDragonData(vertex: vertex, part: 1)
                case .dragonData2: model.showDragonData(vertex: vertex, part: 2)
                }
            case 1: // Move generation
                if self.settings.moveGenMode == .deltaTerritory {
                    DispatchQueue.main.async { self.settings.deltaVertex = vertex }
                    self.refreshMarkup()
                } else {
                    model.showMoveReasons(vertex: vertex)
                }
            case 2: // Eyes
                let color = self.settings.eyeColor == .white ? "white" : "black"
                model.showEyeData(vertex: vertex, color: color)
            case 3: // Influence
                if self.settings.influenceSource == .afterMove
                    || self.settings.influenceSource == .followup {
                    DispatchQueue.main.async { self.settings.moveInfluenceVertex = vertex }
                    self.refreshMarkup()
                }
            case 4: // Reading
                self.handleReadingTap(vertex: vertex, model: model)
            default: break
            }
        }
    }

    private func handleReadingTap(vertex: String, model: RegressionModel) {
        let mode = settings.readingMode
        let needsTwo = mode == .connection || mode == .semeai
            || mode == .owlDoesAttack || mode == .owlDoesDefend

        // Mirror Pike line 1755-1758: clear markup when no first vertex selected
        // or when using single-vertex modes (tactical/owl).
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
            return base + "." + model.name.replacingOccurrences(of: " ", with: "_")
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
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            for model in self.models {
                model.addMarkup(mode: self.currentTabIndex, settings: s)
            }
        }
    }
}

// MARK: - ContentView

struct ContentView: View {
    @EnvironmentObject var state: AppState
    @State private var newTestcaseText: String = ""
    @State private var enginePathText: String = "../../interface/gnugo"
    @State private var engineNameText: String = "Engine 2"

    var body: some View {
        VStack(spacing: 0) {
            // Top: testcase text + engine selector
            HStack(alignment: .top, spacing: 16) {
                ScrollView([.horizontal, .vertical]) {
                    Text(state.fullTestcaseText)
                        .font(.system(size: 11, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(width: 440, height: 90)
                .border(Color.gray.opacity(0.4))

                if state.models.count > 1 {
                    Picker("Engine", selection: $state.currentModelIndex) {
                        ForEach(state.models.indices, id: \.self) { i in
                            Text(state.models[i].name).tag(i)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 200)
                }
            }
            .padding(8)

            Divider()

            HStack(alignment: .top, spacing: 4) {
                // Left: tabbed controls
                TabView(selection: $state.currentTabIndex) {
                    wormsAndDragonsTab.tabItem { Text("worms\n& dragons") }.tag(0)
                    moveGenerationTab.tabItem { Text("move\ngeneration") }.tag(1)
                    eyesTab.tabItem { Text("eyes") }.tag(2)
                    influenceTab.tabItem { Text("influence") }.tag(3)
                    readingTab.tabItem { Text("reading") }.tag(4)
                }
                .frame(width: 260)
                .onChange(of: state.currentTabIndex) { _ in state.refreshMarkup() }

                // Right: board + data
                VStack(spacing: 4) {
                    let model = state.models.indices.contains(state.currentModelIndex)
                        ? state.models[state.currentModelIndex] : nil
                    if let m = model {
                        GobanView(model: m.goban, boardPixelSize: 560) { vertex in
                            state.boardTapped(vertex: vertex, modelIndex: state.currentModelIndex)
                        }

                        Divider()

                        VStack(alignment: .leading, spacing: 2) {
                            Text(m.dataTitle)
                                .font(.system(size: 11, weight: .bold))
                            ScrollView {
                                Text(m.dataText)
                                    .font(.system(size: 10, design: .monospaced))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                        .frame(height: 120)
                        .padding(4)
                    }
                }
            }
            .padding(4)
        }
    }

    // MARK: - Tab pages

    var wormsAndDragonsTab: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Data", selection: $state.settings.wormDragonDataMode) {
                ForEach(WormDragonDataMode.allCases, id: \.self) { m in
                    Text(m.label).tag(m)
                }
            }.pickerStyle(.radioGroup)

            Divider()

            Group {
                radioButton("none", isOn: state.settings.wormDragonOverlay == .none) {
                    state.settings.wormDragonOverlay = .none; state.refreshMarkup()
                }
                radioButton("worm status", isOn: state.settings.wormDragonOverlay == .wormStatus) {
                    state.settings.wormDragonOverlay = .wormStatus; state.refreshMarkup()
                }
                radioButton("dragon status", isOn: state.settings.wormDragonOverlay == .dragonStatus) {
                    state.settings.wormDragonOverlay = .dragonStatus; state.refreshMarkup()
                }
                radioButton("dragon safety", isOn: state.settings.wormDragonOverlay == .dragonSafety) {
                    state.settings.wormDragonOverlay = .dragonSafety; state.refreshMarkup()
                }
            }
            Spacer()
        }
        .padding(8)
    }

    var moveGenerationTab: some View {
        VStack(alignment: .leading, spacing: 4) {
            radioButton("top moves", isOn: state.settings.moveGenMode == .topMoves) {
                state.settings.moveGenMode = .topMoves; state.refreshMarkup()
            }
            radioButton("all moves", isOn: state.settings.moveGenMode == .allMoves) {
                state.settings.moveGenMode = .allMoves; state.refreshMarkup()
            }
            radioButton("delta territory for \(state.settings.deltaVertex)",
                        isOn: state.settings.moveGenMode == .deltaTerritory) {
                state.settings.moveGenMode = .deltaTerritory; state.refreshMarkup()
            }
            Spacer()
        }
        .padding(8)
    }

    var eyesTab: some View {
        VStack(alignment: .leading, spacing: 4) {
            radioButton("white eyes", isOn: state.settings.eyeColor == .white) {
                state.settings.eyeColor = .white; state.refreshMarkup()
            }
            radioButton("black eyes", isOn: state.settings.eyeColor == .black) {
                state.settings.eyeColor = .black; state.refreshMarkup()
            }
            Spacer()
        }
        .padding(8)
    }

    var influenceTab: some View {
        VStack(alignment: .leading, spacing: 4) {
            Group {
                radioButton("white influence, dragons known",
                            isOn: state.settings.influenceSource == .whiteKnown) {
                    state.settings.influenceSource = .whiteKnown; state.refreshMarkup()
                }
                radioButton("black influence, dragons known",
                            isOn: state.settings.influenceSource == .blackKnown) {
                    state.settings.influenceSource = .blackKnown; state.refreshMarkup()
                }
                radioButton("after move influence for \(state.settings.moveInfluenceVertex)",
                            isOn: state.settings.influenceSource == .afterMove) {
                    state.settings.influenceSource = .afterMove; state.refreshMarkup()
                }
                radioButton("followup influence for \(state.settings.moveInfluenceVertex)",
                            isOn: state.settings.influenceSource == .followup) {
                    state.settings.influenceSource = .followup; state.refreshMarkup()
                }
            }
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(InfluenceData.allCases, id: \.label) { d in
                        radioButton(d.label, isOn: state.settings.influenceData == d) {
                            state.settings.influenceData = d; state.refreshMarkup()
                        }
                    }
                }
            }
        }
        .padding(8)
    }

    var readingTab: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach([
                ("tactical reading",    ReadingMode.tactical),
                ("owl reading",         ReadingMode.owl),
                ("owl_does_attack",     ReadingMode.owlDoesAttack),
                ("owl_does_defend",     ReadingMode.owlDoesDefend),
                ("connection reading",  ReadingMode.connection),
                ("semeai reading",      ReadingMode.semeai),
            ], id: \.0) { label, mode in
                radioButton(label, isOn: state.settings.readingMode == mode) {
                    state.settings.readingMode = mode
                    state.settings.firstVertex = nil
                }
            }
            Divider()
            Toggle("save sgf traces to", isOn: $state.settings.saveSgf)
                .onChange(of: state.settings.saveSgf) { v in
                    // Mirror Pike sgf_traces_button_toggled: unchecking save also unchecks viewer
                    if !v { state.settings.openSgfViewer = false }
                }
            TextField("SGF file", text: $state.settings.sgfFile)
                .textFieldStyle(.roundedBorder).font(.system(size: 11))
            Toggle("start sgf viewer as", isOn: $state.settings.openSgfViewer)
                .onChange(of: state.settings.openSgfViewer) { v in
                    // Mirror Pike sgf_viewer_button_toggled: checking viewer also checks save
                    if v { state.settings.saveSgf = true }
                }
            TextField("viewer command", text: $state.settings.sgfViewerCmd)
                .textFieldStyle(.roundedBorder).font(.system(size: 11))
            Divider()
            TextField("load testcase (file:num)", text: $newTestcaseText)
                .textFieldStyle(.roundedBorder).font(.system(size: 11))
                .onSubmit { state.loadNewTestcase(newTestcaseText) }
            Button("Load new testcase") { state.loadNewTestcase(newTestcaseText) }
            if state.testcases.count > 1 {
                HStack {
                    Button("Previous") { state.prevTestcase() }
                        .disabled(state.testcaseIndex == 0)
                    Button("Next") { state.nextTestcase() }
                        .disabled(state.testcaseIndex >= state.testcases.count - 1)
                }
            }
            Divider()
            TextField("engine path", text: $enginePathText)
                .textFieldStyle(.roundedBorder).font(.system(size: 11))
            TextField("engine name", text: $engineNameText)
                .textFieldStyle(.roundedBorder).font(.system(size: 11))
            Button("Start new engine") {
                state.selectNewEngine(path: enginePathText, name: engineNameText)
            }
            Spacer()
        }
        .padding(8)
    }

    // MARK: - Radio button helper

    func radioButton(_ label: String, isOn: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: isOn ? "largecircle.fill.circle" : "circle")
                    .foregroundColor(isOn ? .accentColor : .secondary)
                Text(label).font(.system(size: 12))
                Spacer()
            }
        }
        .buttonStyle(.plain)
    }
}
