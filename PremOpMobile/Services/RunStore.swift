import Foundation

/// One dispatch and everything needed to audit it afterwards.
struct RunRecord: Identifiable, Codable {
    let id: UUID
    let agent: String
    let prompt: String
    var output: String
    var toolCalls: [String]
    var inputTokens: Int
    var outputTokens: Int
    var failed: Bool
    let startedAt: Date

    init(agent: String, prompt: String) {
        id = UUID()
        self.agent = agent
        self.prompt = prompt
        output = ""
        toolCalls = []
        inputTokens = 0
        outputTokens = 0
        failed = false
        startedAt = Date()
    }
}

/// Run history, persisted as JSON in Application Support.
///
/// A file rather than SQLite: the desktop version needs concurrent writers and
/// this does not, so a database would be dependency without benefit.
@MainActor
final class RunStore: ObservableObject {
    @Published private(set) var runs: [RunRecord] = []

    private let url: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base.appendingPathComponent("runs.json")
    }()

    init() { load() }

    func record(_ run: RunRecord) {
        runs.insert(run, at: 0)
        if runs.count > 100 { runs.removeLast(runs.count - 100) }
        save()
    }

    func clear() {
        runs.removeAll()
        save()
    }

    private func load() {
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([RunRecord].self, from: data) else { return }
        runs = decoded
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(runs) else { return }
        try? data.write(to: url, options: .atomic)
    }
}
