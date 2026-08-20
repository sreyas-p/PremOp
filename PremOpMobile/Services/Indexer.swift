import Foundation
import NaturalLanguage

/// Semantic index over everything the agents read on this device.
///
/// Embeddings come from `NLEmbedding`, which ships with iOS — no model to
/// download, no MLX, no network. That is a much better trade here than the
/// 130MB MLX embedder the desktop version uses: it is already on the phone,
/// it is fast, and it costs nothing in app size.
///
/// Storage is a JSON file with brute-force cosine at query time. For a personal
/// corpus that scan is trivial, and it avoids taking a database dependency for
/// something that holds a few thousand short passages.
@MainActor
final class Indexer {
    static let shared = Indexer()

    struct Entry: Codable, Identifiable {
        let id: String          // "\(source):\(sourceID)"
        let source: String
        let sourceID: String
        let title: String
        let text: String
        let vector: [Double]
        let indexedAt: Date
    }

    private var entries: [String: Entry] = [:]
    private let embedding = NLEmbedding.sentenceEmbedding(for: .english)
    private let storeURL: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base.appendingPathComponent("semantic-index.json")
    }()

    private init() { load() }

    var count: Int { entries.count }

    var sourceCounts: [String: Int] {
        entries.values.reduce(into: [:]) { $0[$1.source, default: 0] += 1 }
    }

    /// Index one item, replacing anything previously stored under the same id
    /// so an edited reminder stops matching against its own stale text.
    static func remember(source: String, id: String, title: String, text: String) {
        shared.add(source: source, sourceID: id, title: title, text: text)
    }

    func add(source: String, sourceID: String, title: String, text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let embedding else { return }
        // NLEmbedding works on a single passage; long text is truncated rather
        // than chunked, which is enough for reminders and event titles.
        let passage = String(trimmed.prefix(1_000))
        guard let vector = embedding.vector(for: passage) else { return }

        entries["\(source):\(sourceID)"] = Entry(
            id: "\(source):\(sourceID)", source: source, sourceID: sourceID,
            title: title, text: passage, vector: vector, indexedAt: Date()
        )
        save()
    }

    struct Hit {
        let score: Double
        let entry: Entry
    }

    func search(_ query: String, limit: Int = 8, source: String? = nil) -> [Hit] {
        guard let embedding, let queryVector = embedding.vector(for: query) else { return [] }
        let pool = source.map { s in entries.values.filter { $0.source == s } }
            ?? Array(entries.values)

        return pool
            .map { Hit(score: Self.cosine(queryVector, $0.vector), entry: $0) }
            .sorted { $0.score > $1.score }
            .prefix(limit)
            .map { $0 }
    }

    private static func cosine(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count else { return 0 }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<a.count {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        let denominator = (na.squareRoot() * nb.squareRoot())
        return denominator > 0 ? dot / denominator : 0
    }

    // ── persistence ─────────────────────────────────────────────────────

    private func load() {
        guard let data = try? Data(contentsOf: storeURL),
              let decoded = try? JSONDecoder().decode([Entry].self, from: data) else { return }
        entries = Dictionary(uniqueKeysWithValues: decoded.map { ($0.id, $0) })
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(Array(entries.values)) else { return }
        try? data.write(to: storeURL, options: .atomic)
    }

    func clear() {
        entries.removeAll()
        try? FileManager.default.removeItem(at: storeURL)
    }
}
