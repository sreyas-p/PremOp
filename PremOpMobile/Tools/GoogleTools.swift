import Foundation

/// Gmail, Google Docs, and YouTube — the account-side tools, ported from the
/// desktop version.
///
/// Read-only on Gmail, deliberately: sending mail is outward-facing and
/// irreversible, and a phone in a pocket is the last place it should be
/// possible without an explicit human tap.
@MainActor
enum GoogleTools {
    static let all: [Tool] = [
        gmailSearch, gmailRead, docCreate, docRead, docFind,
        youtubeSearch, youtubeVideo, youtubeLiked
    ]

    // ── Gmail ───────────────────────────────────────────────────────────

    static let gmailSearch = Tool(
        name: "gmail_search",
        description: """
        Search the user's Gmail and return matching message summaries.

        Use Gmail's own query syntax — "from:landlord newer_than:30d",
        "subject:invoice has:attachment", "is:unread". A narrow query is
        dramatically cheaper than a broad one, since every result lands in
        context.

        Returns message ids that gmail_read_message accepts.
        """,
        schema: schema([
            "query": stringProp("A Gmail search query using Gmail's operators."),
            "max_results": intProp("How many messages, 1-25. Defaults to 8.")
        ], required: ["query"]),
        run: { input in
            guard let query = input["query"] as? String else { return "A query is required." }
            let limit = max(1, min(input["max_results"] as? Int ?? 8, 25))
            do {
                let listing = try await GoogleAPI.get(
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages",
                    query: ["q": query, "maxResults": String(limit)],
                    credentials: .workspace)
                let ids = (listing["messages"] as? [[String: Any]] ?? [])
                    .compactMap { $0["id"] as? String }
                guard !ids.isEmpty else { return "No messages matched \(query)." }

                var lines: [String] = ["\(ids.count) message(s):"]
                for id in ids {
                    let message = try await GoogleAPI.get(
                        "https://gmail.googleapis.com/gmail/v1/users/me/messages/\(id)",
                        query: ["format": "metadata",
                                "metadataHeaders": "From,Subject,Date"],
                        credentials: .workspace)
                    let headers = Self.headers(from: message)
                    lines.append(
                        "- id=\(id) | \(headers["date"] ?? "?") | from \(headers["from"] ?? "?")"
                        + " | \(headers["subject"] ?? "(no subject)")"
                        + "\n    \(message["snippet"] as? String ?? "")")
                }
                return lines.joined(separator: "\n")
            } catch {
                return GoogleAPI.describe(error)
            }
        }
    )

    static let gmailRead = Tool(
        name: "gmail_read_message",
        description: "Read the full text of one Gmail message by its id, from gmail_search.",
        schema: schema(["message_id": stringProp("A message id from gmail_search.")],
                       required: ["message_id"]),
        run: { input in
            guard let id = input["message_id"] as? String else { return "A message_id is required." }
            do {
                let message = try await GoogleAPI.get(
                    "https://gmail.googleapis.com/gmail/v1/users/me/messages/\(id)",
                    query: ["format": "full"], credentials: .workspace)
                let headers = Self.headers(from: message)
                let payload = message["payload"] as? [String: Any] ?? [:]
                var body = Self.body(from: payload)
                if body.count > 4_000 { body = String(body.prefix(4_000)) + "\n[truncated]" }

                let subject = headers["subject"] ?? ""
                Indexer.remember(source: "gmail", id: id, title: subject,
                                 text: "From \(headers["from"] ?? "") — \(subject)\n\(body)")
                return """
                From: \(headers["from"] ?? "?")
                Date: \(headers["date"] ?? "?")
                Subject: \(subject)

                \(body.isEmpty ? "[no readable text body]" : body)
                """
            } catch {
                return GoogleAPI.describe(error)
            }
        }
    )

    // ── Google Docs ─────────────────────────────────────────────────────

    static let docCreate = Tool(
        name: "doc_create",
        description: """
        Create a Google Doc and write text into it.

        Unlike a reminder, this syncs to the user's Drive and is the right place
        for anything long. Reminders remain better for short notes they will
        actually see on the phone.
        """,
        schema: schema([
            "title": stringProp("Specific title — 'Lease renewal, Oct 2026', not 'Notes'."),
            "body": stringProp("Plain text contents.")
        ], required: ["title", "body"]),
        run: { input in
            guard let title = input["title"] as? String,
                  let body = input["body"] as? String else {
                return "Both title and body are required."
            }
            do {
                let created = try await GoogleAPI.post(
                    "https://docs.googleapis.com/v1/documents",
                    body: ["title": title], credentials: .workspace)
                guard let id = created["documentId"] as? String else {
                    return "Google did not return a document id."
                }
                _ = try await GoogleAPI.post(
                    "https://docs.googleapis.com/v1/documents/\(id):batchUpdate",
                    body: ["requests": [["insertText": [
                        "location": ["index": 1], "text": body]]]],
                    credentials: .workspace)
                Indexer.remember(source: "doc", id: id, title: title, text: body)
                return "Created '\(title)'\nid: \(id)\nhttps://docs.google.com/document/d/\(id)/edit"
            } catch {
                return GoogleAPI.describe(error)
            }
        }
    )

    static let docRead = Tool(
        name: "doc_read",
        description: """
        Read a Google Doc this app created.

        Only documents created through doc_create are reachable — the app's
        Drive scope covers its own files, not documents the user wrote
        themselves. Asking for one of those cannot work.
        """,
        schema: schema(["document_id": stringProp("An id from doc_create or doc_find.")],
                       required: ["document_id"]),
        run: { input in
            guard let id = input["document_id"] as? String else { return "A document_id is required." }
            do {
                let document = try await GoogleAPI.get(
                    "https://docs.googleapis.com/v1/documents/\(id)", credentials: .workspace)
                let content = (document["body"] as? [String: Any])?["content"] as? [[String: Any]] ?? []
                let text = content.compactMap { block -> String? in
                    guard let paragraph = block["paragraph"] as? [String: Any],
                          let elements = paragraph["elements"] as? [[String: Any]] else { return nil }
                    return elements.compactMap {
                        ($0["textRun"] as? [String: Any])?["content"] as? String
                    }.joined()
                }.joined()

                let title = document["title"] as? String ?? "(untitled)"
                Indexer.remember(source: "doc", id: id, title: title, text: text)
                return "\(title)\n\n\(text.isEmpty ? "[empty]" : text)"
            } catch {
                return GoogleAPI.describe(error)
            }
        }
    )

    static let docFind = Tool(
        name: "doc_find",
        description: "Find Google Docs this app created, by title. Titles only — contents are not searched.",
        schema: schema(["query": stringProp("Phrase to match against titles.")],
                       required: ["query"]),
        run: { input in
            guard let query = input["query"] as? String else { return "A query is required." }
            // Drive query strings are single-quoted, so an apostrophe would
            // terminate the literal and the API rejects the whole thing.
            let escaped = query.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "'", with: "\\'")
            do {
                let results = try await GoogleAPI.get(
                    "https://www.googleapis.com/drive/v3/files",
                    query: ["q": "mimeType='application/vnd.google-apps.document' and name contains '\(escaped)' and trashed=false",
                            "fields": "files(id,name,modifiedTime)",
                            "orderBy": "modifiedTime desc", "pageSize": "10"],
                    credentials: .workspace)
                let files = results["files"] as? [[String: Any]] ?? []
                guard !files.isEmpty else { return "No documents matched \(query)." }
                return files.map {
                    "- \($0["name"] as? String ?? "?") (id=\($0["id"] as? String ?? "?"))"
                }.joined(separator: "\n")
            } catch {
                return GoogleAPI.describe(error)
            }
        }
    )

    // ── YouTube ─────────────────────────────────────────────────────────

    static let youtubeSearch = Tool(
        name: "youtube_search",
        description: "Search all of public YouTube for videos. Not the user's own data.",
        schema: schema([
            "query": stringProp("What to search for."),
            "max_results": intProp("How many, 1-25. Defaults to 8.")
        ], required: ["query"]),
        run: { input in
            guard let query = input["query"] as? String else { return "A query is required." }
            do {
                let response = try await GoogleAPI.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    query: ["part": "snippet", "q": query, "type": "video",
                            "maxResults": String(max(1, min(input["max_results"] as? Int ?? 8, 25)))],
                    credentials: .youtube)
                return Self.renderVideos(response["items"] as? [[String: Any]] ?? [], label: query)
            } catch {
                return GoogleAPI.describe(error)
            }
        }
    )

    static let youtubeVideo = Tool(
        name: "youtube_video_details",
        description: """
        Get a video's title, channel, description, and statistics.

        Transcripts are not available through the API for videos the user does
        not own, so this metadata is what there is — never guess at what a video
        says beyond it.
        """,
        schema: schema(["video_id": stringProp("A YouTube video id.")], required: ["video_id"]),
        run: { input in
            guard let id = input["video_id"] as? String else { return "A video_id is required." }
            do {
                let response = try await GoogleAPI.get(
                    "https://www.googleapis.com/youtube/v3/videos",
                    query: ["part": "snippet,statistics", "id": id], credentials: .youtube)
                guard let video = (response["items"] as? [[String: Any]])?.first,
                      let snippet = video["snippet"] as? [String: Any] else {
                    return "No video found with id \(id)."
                }
                let stats = video["statistics"] as? [String: Any] ?? [:]
                var description = snippet["description"] as? String ?? ""
                if description.count > 1_500 { description = String(description.prefix(1_500)) + " […]" }

                let title = snippet["title"] as? String ?? ""
                Indexer.remember(source: "youtube", id: id, title: title,
                                 text: "\(title) by \(snippet["channelTitle"] as? String ?? ""). \(description)")
                return """
                Title: \(title)
                Channel: \(snippet["channelTitle"] as? String ?? "?")
                Published: \(snippet["publishedAt"] as? String ?? "?")
                Views: \(stats["viewCount"] as? String ?? "?") | Likes: \(stats["likeCount"] as? String ?? "?")
                https://www.youtube.com/watch?v=\(id)

                \(description)
                """
            } catch {
                return GoogleAPI.describe(error)
            }
        }
    )

    static let youtubeLiked = Tool(
        name: "youtube_liked_videos",
        description: """
        List videos the user has liked, newest first.

        Watch history is not retrievable through the API at all — this is the
        closest available signal for what they have been watching. Do not
        describe it as watch history.
        """,
        schema: schema(["max_results": intProp("How many, 1-25. Defaults to 10.")]),
        run: { input in
            do {
                let response = try await GoogleAPI.get(
                    "https://www.googleapis.com/youtube/v3/playlistItems",
                    query: ["part": "snippet", "playlistId": "LL",
                            "maxResults": String(max(1, min(input["max_results"] as? Int ?? 10, 25)))],
                    credentials: .youtube)
                return Self.renderVideos(response["items"] as? [[String: Any]] ?? [],
                                         label: "liked videos")
            } catch {
                return GoogleAPI.describe(error)
            }
        }
    )

    // ── helpers ─────────────────────────────────────────────────────────

    private static func headers(from message: [String: Any]) -> [String: String] {
        let payload = message["payload"] as? [String: Any] ?? [:]
        let list = payload["headers"] as? [[String: Any]] ?? []
        var out: [String: String] = [:]
        for header in list {
            if let name = (header["name"] as? String)?.lowercased(),
               let value = header["value"] as? String {
                out[name] = value
            }
        }
        return out
    }

    /// Walks the MIME tree for the best text representation available.
    private static func body(from payload: [String: Any]) -> String {
        func decode(_ encoded: String) -> String {
            var padded = encoded.replacingOccurrences(of: "-", with: "+")
                .replacingOccurrences(of: "_", with: "/")
            while padded.count % 4 != 0 { padded += "=" }
            guard let data = Data(base64Encoded: padded) else { return "" }
            return String(data: data, encoding: .utf8) ?? ""
        }

        if let data = (payload["body"] as? [String: Any])?["data"] as? String,
           payload["mimeType"] as? String == "text/plain" {
            return decode(data)
        }
        let parts = payload["parts"] as? [[String: Any]] ?? []
        for wanted in ["text/plain", "text/html"] {
            for part in parts where part["mimeType"] as? String == wanted {
                if let data = (part["body"] as? [String: Any])?["data"] as? String {
                    return decode(data)
                }
            }
        }
        for part in parts {
            let nested = body(from: part)
            if !nested.isEmpty { return nested }
        }
        if let data = (payload["body"] as? [String: Any])?["data"] as? String {
            return decode(data)
        }
        return ""
    }

    private static func renderVideos(_ items: [[String: Any]], label: String) -> String {
        guard !items.isEmpty else { return "Nothing found for \(label)." }
        return "\(items.count) result(s) for \(label):\n" + items.compactMap { item -> String? in
            guard let snippet = item["snippet"] as? [String: Any] else { return nil }
            let id = (snippet["resourceId"] as? [String: Any])?["videoId"] as? String
                ?? (item["id"] as? [String: Any])?["videoId"] as? String ?? "?"
            return "- \(snippet["title"] as? String ?? "?") (id=\(id))"
                + "\n    \(snippet["videoOwnerChannelTitle"] as? String ?? snippet["channelTitle"] as? String ?? "?")"
        }.joined(separator: "\n")
    }
}
