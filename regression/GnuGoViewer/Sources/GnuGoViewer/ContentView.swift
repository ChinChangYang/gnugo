import SwiftUI

struct ContentView: View {
    @Environment(AppState.self) var state

    var body: some View {
        @Bindable var state = state
        return VStack(spacing: 0) {
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
                TabView(selection: $state.currentTab) {
                    WormsAndDragonsTab().tabItem { Text("worms\n& dragons") }.tag(TabSelection.wormsAndDragons)
                    MoveGenerationTab().tabItem { Text("move\ngeneration") }.tag(TabSelection.moveGeneration)
                    EyesTab().tabItem { Text("eyes") }.tag(TabSelection.eyes)
                    InfluenceTab().tabItem { Text("influence") }.tag(TabSelection.influence)
                    ReadingTab().tabItem { Text("reading") }.tag(TabSelection.reading)
                }
                .frame(width: 260)
                .onChange(of: state.currentTab) { state.refreshMarkup() }

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
            .onAppear {
                NSApp.activate(ignoringOtherApps: true)
            }
        }
    }
}
