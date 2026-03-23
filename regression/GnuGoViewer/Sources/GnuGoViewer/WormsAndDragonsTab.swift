import SwiftUI

struct WormsAndDragonsTab: View {
    @Environment(AppState.self) var state

    var body: some View {
        @Bindable var state = state
        return VStack(alignment: .leading, spacing: 8) {
            Picker("Data", selection: $state.settings.wormDragonDataMode) {
                ForEach(WormDragonDataMode.allCases, id: \.self) { m in
                    Text(m.label).tag(m)
                }
            }.pickerStyle(.radioGroup)

            Divider()

            Group {
                RadioButton(label: "none", isOn: state.settings.wormDragonOverlay == .none) {
                    state.settings.wormDragonOverlay = .none; state.refreshMarkup()
                }
                RadioButton(label: "worm status", isOn: state.settings.wormDragonOverlay == .wormStatus) {
                    state.settings.wormDragonOverlay = .wormStatus; state.refreshMarkup()
                }
                RadioButton(label: "dragon status", isOn: state.settings.wormDragonOverlay == .dragonStatus) {
                    state.settings.wormDragonOverlay = .dragonStatus; state.refreshMarkup()
                }
                RadioButton(label: "dragon safety", isOn: state.settings.wormDragonOverlay == .dragonSafety) {
                    state.settings.wormDragonOverlay = .dragonSafety; state.refreshMarkup()
                }
            }
            Spacer()
        }
        .padding(8)
    }
}
