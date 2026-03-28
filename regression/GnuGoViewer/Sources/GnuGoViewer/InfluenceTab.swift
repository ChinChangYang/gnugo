import SwiftUI

struct InfluenceTab: View {
    @Environment(AppState.self) var state

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Group {
                RadioButton(label: "white influence, dragons known",
                            isOn: state.settings.influenceSource == .whiteKnown) {
                    state.settings.influenceSource = .whiteKnown; state.refreshMarkup()
                }
                RadioButton(label: "black influence, dragons known",
                            isOn: state.settings.influenceSource == .blackKnown) {
                    state.settings.influenceSource = .blackKnown; state.refreshMarkup()
                }
                RadioButton(label: "after move influence for \(state.settings.moveInfluenceVertex)",
                            isOn: state.settings.influenceSource == .afterMove) {
                    state.settings.influenceSource = .afterMove; state.refreshMarkup()
                }
                RadioButton(label: "followup influence for \(state.settings.moveInfluenceVertex)",
                            isOn: state.settings.influenceSource == .followup) {
                    state.settings.influenceSource = .followup; state.refreshMarkup()
                }
            }
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(InfluenceData.allCases, id: \.label) { d in
                        RadioButton(label: d.label, isOn: state.settings.influenceData == d) {
                            state.settings.influenceData = d; state.refreshMarkup()
                        }
                    }
                }
            }
        }
        .padding(8)
    }
}
