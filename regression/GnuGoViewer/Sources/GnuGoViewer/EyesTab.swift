import SwiftUI

struct EyesTab: View {
    @Environment(AppState.self) var state

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            RadioButton(label: "white eyes", isOn: state.settings.eyeColor == .white) {
                state.settings.eyeColor = .white; state.refreshMarkup()
            }
            RadioButton(label: "black eyes", isOn: state.settings.eyeColor == .black) {
                state.settings.eyeColor = .black; state.refreshMarkup()
            }
            Spacer()
        }
        .padding(8)
    }
}
