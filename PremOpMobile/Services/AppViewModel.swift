import Foundation
import SwiftUI

@MainActor
final class AppViewModel: ObservableObject {
    @Published var prompt = ""
    @Published var selectedAgent = "dispatcher"
    @Published var output = ""
    @Published var progress: [String] = []
    @Published var isRunning = false
    @Published var lastRun: RunRecord?
    @Published var apiKeyDraft = ""
    @Published var googleClientDraft = ""
    @Published var googleStatus: [String: Bool] = [:]
    @Published var googleError: String?

    let store = RunStore()

    var agents: [AgentSpec] { Agents.all }
    var hasKey: Bool { KeychainStore.hasKey }
    var indexedCount: Int { Indexer.shared.count }

    func loadKeyDraft() {
        apiKeyDraft = KeychainStore.apiKey ?? ""
        googleClientDraft = KeychainStore.googleClientID ?? ""
        refreshGoogleStatus()
    }

    func refreshGoogleStatus() {
        googleStatus = Dictionary(uniqueKeysWithValues: GoogleCredentialSet.all.map {
            ($0.name, GoogleAuth.shared.isConnected($0))
        })
    }

    func saveGoogleClientID() {
        KeychainStore.googleClientID = googleClientDraft
        objectWillChange.send()
    }

    /// Both consents are separate because drive.file and youtube.readonly
    /// cannot be granted in one request — Google's constraint, not ours.
    func connectGoogle(_ set: GoogleCredentialSet) async {
        googleError = nil
        do {
            try await GoogleAuth.shared.connect(set)
        } catch {
            googleError = error.localizedDescription
        }
        refreshGoogleStatus()
    }

    func disconnectGoogle(_ set: GoogleCredentialSet) {
        GoogleAuth.shared.disconnect(set)
        refreshGoogleStatus()
    }

    func saveKey() {
        KeychainStore.apiKey = apiKeyDraft
        objectWillChange.send()
    }

    func run() async {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isRunning else { return }
        guard let spec = Agents.named(selectedAgent) else { return }

        isRunning = true
        progress = []
        output = ""
        var record = RunRecord(agent: spec.name, prompt: text)

        let (tools, missing) = ToolRegistry.resolve(spec.toolNames)
        if !missing.isEmpty {
            progress.append("⚠︎ unregistered tools skipped: \(missing.joined(separator: ", "))")
        }

        do {
            let result = try await AnthropicClient().run(
                system: spec.system, prompt: text, tools: tools, maxTokens: spec.maxTokens
            ) { [weak self] line in
                Task { @MainActor in self?.progress.append(line) }
            }
            output = result.text
            record.output = result.text
            record.toolCalls = result.toolCalls
            record.inputTokens = result.usage.inputTokens
            record.outputTokens = result.usage.outputTokens
        } catch {
            output = error.localizedDescription
            record.output = error.localizedDescription
            record.failed = true
        }

        store.record(record)
        lastRun = record
        isRunning = false
    }
}
