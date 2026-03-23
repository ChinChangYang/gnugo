import SwiftUI

struct ContentView: View {
    @Environment(AppState.self) var state

    private var model: RegressionModel? {
        state.models.indices.contains(state.currentModelIndex)
            ? state.models[state.currentModelIndex] : nil
    }

    var body: some View {
        @Bindable var state = state
        let model = self.model
        return HStack(spacing: 0) {
            // Left: sidebar with test info and tabs
            VStack(spacing: 0) {
                // Test case info
                VStack(alignment: .leading, spacing: 4) {
                    ScrollView([.horizontal, .vertical]) {
                        Text(state.fullTestcaseText)
                            .font(.system(.body, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(height: 70)
                    .border(Color.gray.opacity(0.4))

                    if state.models.count > 1 {
                        Picker("Engine", selection: $state.currentModelIndex) {
                            ForEach(state.models.indices, id: \.self) { i in
                                Text(state.models[i].name).tag(i)
                            }
                        }
                        .pickerStyle(.menu)
                    }
                }
                .padding(8)

                Divider()

                // Tabbed controls
                TabView(selection: $state.currentTab) {
                    WormsAndDragonsTab().tabItem { Text("worms") }.tag(TabSelection.wormsAndDragons)
                    MoveGenerationTab().tabItem { Text("moves") }.tag(TabSelection.moveGeneration)
                    EyesTab().tabItem { Text("eyes") }.tag(TabSelection.eyes)
                    InfluenceTab().tabItem { Text("influence") }.tag(TabSelection.influence)
                    ReadingTab().tabItem { Text("reading") }.tag(TabSelection.reading)
                }
                .layoutPriority(1)
                .onChange(of: state.currentTab) { state.refreshMarkup() }
            }
            .frame(width: 280)

            Divider()

            // Center: board fills remaining space
            GeometryReader { geo in
                let boardSize = floor(min(geo.size.width, geo.size.height))
                if let m = model {
                    GobanView(model: m.goban, boardPixelSize: boardSize) { vertex in
                        state.boardTapped(vertex: vertex, modelIndex: state.currentModelIndex)
                    }
                    .frame(width: boardSize, height: boardSize)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }

            Divider()

            // Right: data results panel
            VStack(alignment: .leading, spacing: 2) {
                if let m = model {
                    Text(m.dataTitle)
                        .font(.body.bold())
                    ScrollView {
                        Text(m.dataText)
                            .font(.system(.body, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            .padding(8)
            .frame(width: 250)
        }
        .onAppear {
            NSApp.activate()
        }
    }
}
