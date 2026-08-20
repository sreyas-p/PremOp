import Foundation

/// A thin authorized JSON client for the three Google APIs used here.
///
/// Hand-rolled rather than taking the official SDK: this needs four endpoints,
/// and the SDK brings a dependency graph far larger than the surface it would
/// cover.
enum GoogleAPI {
    struct APIError: LocalizedError {
        let status: Int
        let body: String
        var errorDescription: String? {
            if status == 403 {
                return "Google refused the request (403). The API may not be enabled for this project, or the scope was not granted: \(body.prefix(160))"
            }
            return "Google returned \(status): \(body.prefix(200))"
        }
    }

    static func get(_ path: String, query: [String: String] = [:],
                    credentials: GoogleCredentialSet) async throws -> [String: Any] {
        var components = URLComponents(string: path)!
        if !query.isEmpty {
            components.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) }
        }
        var request = URLRequest(url: components.url!)
        request.timeoutInterval = 60
        return try await send(request, credentials: credentials)
    }

    static func post(_ path: String, body: [String: Any],
                     credentials: GoogleCredentialSet) async throws -> [String: Any] {
        var request = URLRequest(url: URL(string: path)!)
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        return try await send(request, credentials: credentials)
    }

    private static func send(_ request: URLRequest,
                             credentials: GoogleCredentialSet) async throws -> [String: Any] {
        var request = request
        let token = try await GoogleAuth.shared.accessToken(for: credentials)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            throw APIError(status: status, body: String(data: data, encoding: .utf8) ?? "")
        }
        return (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }

    /// Turns any thrown error into something an agent can act on rather than
    /// a stack trace it will try to reason about.
    static func describe(_ error: Error) -> String {
        if let auth = error as? GoogleAuthError { return auth.localizedDescription }
        if let api = error as? APIError { return api.localizedDescription }
        return "Request failed: \(error.localizedDescription)"
    }
}
