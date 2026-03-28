import Foundation

// MARK: - Tab selection

enum TabSelection: Int, CaseIterable {
    case wormsAndDragons, moveGeneration, eyes, influence, reading
}

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

    var gtpCommand: String {
        switch self {
        case .regions:           return "influence_regions"
        case .territoryValue:    return "territory_value"
        case .whiteInfluence:    return "white_influence"
        case .blackInfluence:    return "black_influence"
        case .whiteStrength:     return "white_strength"
        case .blackStrength:     return "black_strength"
        case .whitePermeability: return "white_permeability"
        case .blackPermeability: return "black_permeability"
        case .whiteAttenuation:  return "white_attenuation"
        case .blackAttenuation:  return "black_attenuation"
        case .nonTerritory:      return "non_territory"
        }
    }
}

enum ReadingMode {
    case tactical, owl, owlDoesAttack, owlDoesDefend, connection, semeai

    var needsTwoVertices: Bool {
        switch self {
        case .connection, .semeai, .owlDoesAttack, .owlDoesDefend: return true
        case .tactical, .owl: return false
        }
    }
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
