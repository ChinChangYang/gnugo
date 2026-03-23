import SwiftUI

struct MoveGenerationTab: View {
    @Environment(AppState.self) var state

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            RadioButton(label: "top moves", isOn: state.settings.moveGenMode == .topMoves) {
                state.settings.moveGenMode = .topMoves; state.refreshMarkup()
            }
            RadioButton(label: "all moves", isOn: state.settings.moveGenMode == .allMoves) {
                state.settings.moveGenMode = .allMoves; state.refreshMarkup()
            }
            RadioButton(label: "delta territory for \(state.settings.deltaVertex)",
                        isOn: state.settings.moveGenMode == .deltaTerritory) {
                state.settings.moveGenMode = .deltaTerritory; state.refreshMarkup()
            }
            Spacer()
        }
        .padding(8)
    }
}
