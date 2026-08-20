import Foundation

@MainActor
enum SearchTools {
    static let all: [Tool] = [semanticSearch]

    static let semanticSearch = Tool(
        name: "semantic_search",
        description: """
        Search previously-seen content by meaning rather than exact words — \
        events, reminders, and notes that have already been read on this device.

        It only covers what has already been looked at through these tools. It \
        is not a search of the whole device. If nothing matches, list the \
        relevant source first (calendar_events, reminders_list) and search \
        again afterwards, since reading indexes.
        """,
        schema: schema([
            "query": stringProp("What you are looking for, in natural language. Full phrases work better than keywords."),
            "limit": intProp("Maximum results, 1-20. Defaults to 6."),
            "source": stringProp("Optional filter: 'calendar' or 'reminder'.")
        ], required: ["query"]),
        run: { input in
            guard let query = input["query"] as? String, !query.isEmpty else {
                return "A query is required."
            }
            let limit = max(1, min(input["limit"] as? Int ?? 6, 20))
            let source = (input["source"] as? String).flatMap { $0.isEmpty ? nil : $0 }
            let hits = Indexer.shared.search(query, limit: limit, source: source)

            guard !hits.isEmpty else {
                let total = Indexer.shared.count
                return total == 0
                    ? "Nothing has been indexed yet. Read something first — calendar_events or reminders_list — then search again."
                    : "No match for \(query) among \(total) indexed item(s)."
            }
            return "\(hits.count) match(es):\n" + hits.map {
                "[\(String(format: "%.2f", $0.score))] \($0.entry.source) — \($0.entry.title)\n    \($0.entry.text.prefix(200))"
            }.joined(separator: "\n")
        }
    )
}
