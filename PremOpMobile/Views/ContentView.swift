import SwiftUI

struct ContentView: View {
    @StateObject private var model = AppViewModel()

    var body: some View {
        TabView {
            RunView(model: model)
                .tabItem { Label("Run", systemImage: "sparkles") }
            HistoryView(store: model.store)
                .tabItem { Label("History", systemImage: "clock") }
            SettingsView(model: model)
                .tabItem { Label("Settings", systemImage: "gearshape") }
        }
        .tint(.indigo)
    }
}

struct RunView: View {
    @ObservedObject var model: AppViewModel
    @FocusState private var promptFocused: Bool

    private let examples = [
        "What's on my calendar this week?",
        "How did I sleep over the last 7 days?",
        "Note down that the lease renews in October, then tell me what's on Friday.",
        "Who is Alex in my contacts, and do I have anything scheduled with them?"
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if !model.hasKey {
                        Label("No API key set — add one in Settings.", systemImage: "exclamationmark.triangle")
                            .font(.footnote)
                            .foregroundStyle(.orange)
                    }

                    Picker("Agent", selection: $model.selectedAgent) {
                        ForEach(model.agents) { Text($0.name).tag($0.name) }
                    }
                    .pickerStyle(.segmented)

                    if let spec = Agents.named(model.selectedAgent) {
                        Text(spec.summary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    TextField("What should it do?", text: $model.prompt, axis: .vertical)
                        .lineLimit(3...8)
                        .textFieldStyle(.roundedBorder)
                        .focused($promptFocused)

                    if model.prompt.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            ForEach(examples, id: \.self) { example in
                                Button(example) { model.prompt = example }
                                    .font(.caption)
                                    .buttonStyle(.plain)
                                    .foregroundStyle(.indigo)
                                    .multilineTextAlignment(.leading)
                            }
                        }
                    }

                    Button {
                        promptFocused = false
                        Task { await model.run() }
                    } label: {
                        if model.isRunning {
                            HStack { ProgressView(); Text("Working…") }
                        } else {
                            Text("Run").frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.isRunning || model.prompt.isEmpty)

                    if !model.progress.isEmpty {
                        Text(model.progress.joined(separator: "  "))
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }

                    if !model.output.isEmpty {
                        Text(model.output)
                            .font(.callout)
                            .textSelection(.enabled)
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(.secondarySystemBackground), in: .rect(cornerRadius: 10))
                    }

                    if let run = model.lastRun, !run.toolCalls.isEmpty {
                        Text("tools: \(run.toolCalls.joined(separator: ", "))\n\(run.inputTokens) in / \(run.outputTokens) out")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.tertiary)
                    }
                }
                .padding()
            }
            .navigationTitle("PremOp")
        }
    }
}

struct HistoryView: View {
    @ObservedObject var store: RunStore

    var body: some View {
        NavigationStack {
            List(store.runs) { run in
                NavigationLink {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                            Text(run.prompt).font(.callout.weight(.medium))
                            Divider()
                            Text(run.output).font(.callout).textSelection(.enabled)
                            Text("\(run.agent) · \(run.inputTokens) in / \(run.outputTokens) out\ntools: \(run.toolCalls.isEmpty ? "—" : run.toolCalls.joined(separator: ", "))")
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                    }
                    .navigationTitle(run.agent)
                } label: {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(run.prompt).lineLimit(1).font(.callout)
                        Text("\(run.agent) · \(run.startedAt.formatted(date: .abbreviated, time: .shortened))")
                            .font(.caption2)
                            .foregroundStyle(run.failed ? .red : .secondary)
                    }
                }
            }
            .overlay {
                if store.runs.isEmpty {
                    ContentUnavailableView("No runs yet", systemImage: "clock")
                }
            }
            .navigationTitle("History")
        }
    }
}

struct SettingsView: View {
    @ObservedObject var model: AppViewModel

    var body: some View {
        NavigationStack {
            Form {
                Section("Anthropic") {
                    SecureField("sk-ant-…", text: $model.apiKeyDraft)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Button("Save key") { model.saveKey() }
                        .disabled(model.apiKeyDraft.isEmpty)
                    Text("Stored in the iOS keychain, never in the app bundle.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                Section("On-device index") {
                    LabeledContent("Indexed items", value: "\(model.indexedCount)")
                    Text("Fills as agents read your calendar and reminders. Embeddings come from iOS's Natural Language framework — nothing is sent anywhere to build it.")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("Clear index", role: .destructive) { Indexer.shared.clear() }
                }

                Section("What this app can reach") {
                    capability("Calendar", detail: "read and write", ok: true)
                    capability("Reminders", detail: "read and write — where notes go", ok: true)
                    capability("Contacts", detail: "read only", ok: true)
                    capability("Photos", detail: "metadata only, never contents", ok: true)
                    capability("Health", detail: "read only", ok: true)
                    capability("Apple Notes", detail: "no API exists on iOS", ok: false)
                    capability("Mail / Messages", detail: "no read API on iOS", ok: false)
                }

                Section("Agents") {
                    ForEach(Agents.all) { spec in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(spec.name).font(.callout.weight(.medium))
                            Text(spec.summary).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Settings")
            .onAppear { model.loadKeyDraft() }
        }
    }

    private func capability(_ name: String, detail: String, ok: Bool) -> some View {
        HStack {
            Image(systemName: ok ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundStyle(ok ? .green : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(name).font(.callout)
                Text(detail).font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}
