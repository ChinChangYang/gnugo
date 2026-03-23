import SwiftUI

struct ReadingTab: View {
    @Environment(AppState.self) var state
    @State private var newTestcaseText: String = ""
    @State private var enginePathText: String = "../../interface/gnugo"
    @State private var engineNameText: String = "Engine 2"

    var body: some View {
        @Bindable var state = state
        return VStack(alignment: .leading, spacing: 4) {
            ForEach([
                ("tactical reading",    ReadingMode.tactical),
                ("owl reading",         ReadingMode.owl),
                ("owl_does_attack",     ReadingMode.owlDoesAttack),
                ("owl_does_defend",     ReadingMode.owlDoesDefend),
                ("connection reading",  ReadingMode.connection),
                ("semeai reading",      ReadingMode.semeai),
            ], id: \.0) { label, mode in
                RadioButton(label: label, isOn: state.settings.readingMode == mode) {
                    state.settings.readingMode = mode
                    state.settings.firstVertex = nil
                }
            }
            Divider()
            Toggle("save sgf traces to", isOn: $state.settings.saveSgf)
                .onChange(of: state.settings.saveSgf) { _, newValue in
                    if !newValue { state.settings.openSgfViewer = false }
                }
            TextField("SGF file", text: $state.settings.sgfFile)
                .textFieldStyle(.roundedBorder).font(.system(size: 11))
            Toggle("start sgf viewer as", isOn: $state.settings.openSgfViewer)
                .onChange(of: state.settings.openSgfViewer) { _, newValue in
                    if newValue { state.settings.saveSgf = true }
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
}
