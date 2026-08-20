import AuthenticationServices
import CryptoKit
import Foundation

/// Google OAuth for a public client, which is what an iOS app is.
///
/// Three things differ from the desktop flow and each is load-bearing:
///
/// - **No client secret.** iOS clients cannot keep one, so the exchange is
///   authenticated with PKCE instead. A secret shipped in an app bundle is not
///   a secret.
/// - **No loopback server.** The desktop version listens on `localhost:0` and
///   lets the browser redirect back. iOS cannot, so the redirect is a custom
///   scheme — the reversed client ID — intercepted by
///   `ASWebAuthenticationSession`.
/// - **Two consents, still.** `drive.file` and `youtube.readonly` cannot be
///   granted in one request; that is Google's constraint, not ours, and it
///   applies here exactly as it did on the desktop.
enum GoogleScope {
    static let gmail = "https://www.googleapis.com/auth/gmail.readonly"
    static let documents = "https://www.googleapis.com/auth/documents"
    static let drive = "https://www.googleapis.com/auth/drive.file"
    static let youtube = "https://www.googleapis.com/auth/youtube.readonly"
}

struct GoogleCredentialSet: Sendable, Hashable {
    let name: String
    let scopes: [String]

    var storageKey: String { "google-token-\(name)" }

    /// Gmail, Docs, and Drive together — the grouping proven to work.
    static let workspace = GoogleCredentialSet(
        name: "workspace",
        scopes: [GoogleScope.gmail, GoogleScope.documents, GoogleScope.drive]
    )

    /// YouTube alone, because it conflicts with `drive.file`.
    static let youtube = GoogleCredentialSet(
        name: "youtube", scopes: [GoogleScope.youtube]
    )

    static let all: [GoogleCredentialSet] = [.workspace, .youtube]
}

private struct StoredToken: Codable {
    var accessToken: String
    var refreshToken: String
    var expiresAt: Date

    var isFresh: Bool { expiresAt.timeIntervalSinceNow > 60 }
}

enum GoogleAuthError: LocalizedError {
    case noClientID
    case notConnected(String)
    case cancelled
    case server(String)

    var errorDescription: String? {
        switch self {
        case .noClientID:
            return "No Google client ID. Add one in Settings — create an iOS OAuth client in the Google Cloud console."
        case let .notConnected(name):
            return "Google '\(name)' is not connected. Connect it in Settings."
        case .cancelled:
            return "Sign-in was cancelled."
        case let .server(detail):
            return "Google rejected the request: \(detail)"
        }
    }
}

@MainActor
final class GoogleAuth: NSObject, ASWebAuthenticationPresentationContextProviding {
    static let shared = GoogleAuth()

    private var sessions: [String: ASWebAuthenticationSession] = [:]

    // ── configuration ───────────────────────────────────────────────────

    /// The reversed client ID, which is also the redirect scheme Google
    /// requires for iOS clients: `com.googleusercontent.apps.123-abc`.
    private func redirectScheme(for clientID: String) -> String {
        let bare = clientID.replacingOccurrences(of: ".apps.googleusercontent.com", with: "")
        return "com.googleusercontent.apps.\(bare)"
    }

    func isConnected(_ set: GoogleCredentialSet) -> Bool {
        load(set) != nil
    }

    func disconnect(_ set: GoogleCredentialSet) {
        KeychainStore.setValue(nil, for: set.storageKey)
    }

    // ── consent ─────────────────────────────────────────────────────────

    func connect(_ set: GoogleCredentialSet) async throws {
        guard let clientID = KeychainStore.googleClientID, !clientID.isEmpty else {
            throw GoogleAuthError.noClientID
        }
        let scheme = redirectScheme(for: clientID)
        let redirect = "\(scheme):/oauth2redirect"

        let verifier = Self.randomVerifier()
        let challenge = Self.challenge(for: verifier)

        var components = URLComponents(string: "https://accounts.google.com/o/oauth2/v2/auth")!
        components.queryItems = [
            .init(name: "client_id", value: clientID),
            .init(name: "redirect_uri", value: redirect),
            .init(name: "response_type", value: "code"),
            .init(name: "scope", value: set.scopes.joined(separator: " ")),
            .init(name: "code_challenge", value: challenge),
            .init(name: "code_challenge_method", value: "S256"),
            // Without both of these Google returns no refresh token on repeat
            // consents, and the connection silently dies in an hour.
            .init(name: "access_type", value: "offline"),
            .init(name: "prompt", value: "consent")
        ]

        let callback: URL = try await withCheckedThrowingContinuation { continuation in
            let session = ASWebAuthenticationSession(
                url: components.url!, callbackURLScheme: scheme
            ) { url, error in
                if let url {
                    continuation.resume(returning: url)
                } else if let error = error as? ASWebAuthenticationSessionError,
                          error.code == .canceledLogin {
                    continuation.resume(throwing: GoogleAuthError.cancelled)
                } else {
                    continuation.resume(
                        throwing: error ?? GoogleAuthError.server("no callback"))
                }
            }
            session.presentationContextProvider = self
            session.prefersEphemeralWebBrowserSession = false
            sessions[set.name] = session
            session.start()
        }
        sessions[set.name] = nil

        guard let code = URLComponents(url: callback, resolvingAgainstBaseURL: false)?
            .queryItems?.first(where: { $0.name == "code" })?.value else {
            throw GoogleAuthError.server("no authorization code in callback")
        }

        let token = try await exchange(
            code: code, verifier: verifier, clientID: clientID, redirect: redirect)
        save(token, for: set)
    }

    // ── tokens ──────────────────────────────────────────────────────────

    /// A usable access token, refreshed if expired.
    func accessToken(for set: GoogleCredentialSet) async throws -> String {
        guard let stored = load(set) else { throw GoogleAuthError.notConnected(set.name) }
        if stored.isFresh { return stored.accessToken }

        guard let clientID = KeychainStore.googleClientID else {
            throw GoogleAuthError.noClientID
        }
        let refreshed = try await refresh(stored, clientID: clientID)
        save(refreshed, for: set)
        return refreshed.accessToken
    }

    private func exchange(code: String, verifier: String,
                          clientID: String, redirect: String) async throws -> StoredToken {
        let body = [
            "code": code, "client_id": clientID, "code_verifier": verifier,
            "grant_type": "authorization_code", "redirect_uri": redirect
        ]
        let json = try await post(body)
        guard let access = json["access_token"] as? String,
              let refresh = json["refresh_token"] as? String else {
            throw GoogleAuthError.server(
                "no refresh token returned — the app may already be authorized; "
                + "revoke it at myaccount.google.com and try again")
        }
        return StoredToken(
            accessToken: access, refreshToken: refresh,
            expiresAt: Date().addingTimeInterval(json["expires_in"] as? Double ?? 3_600)
        )
    }

    private func refresh(_ token: StoredToken, clientID: String) async throws -> StoredToken {
        let json = try await post([
            "client_id": clientID, "refresh_token": token.refreshToken,
            "grant_type": "refresh_token"
        ])
        guard let access = json["access_token"] as? String else {
            throw GoogleAuthError.server("refresh failed — reconnect in Settings")
        }
        var updated = token
        updated.accessToken = access
        updated.expiresAt = Date().addingTimeInterval(json["expires_in"] as? Double ?? 3_600)
        return updated
    }

    private func post(_ fields: [String: String]) async throws -> [String: Any] {
        var request = URLRequest(url: URL(string: "https://oauth2.googleapis.com/token")!)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpBody = fields
            .map { "\($0.key)=\($0.value.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? $0.value)" }
            .joined(separator: "&")
            .data(using: .utf8)

        let (data, response) = try await URLSession.shared.data(for: request)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200,
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw GoogleAuthError.server(String(data: data, encoding: .utf8)?.prefix(200).description ?? "\(code)")
        }
        return json
    }

    // ── storage ─────────────────────────────────────────────────────────

    private func load(_ set: GoogleCredentialSet) -> StoredToken? {
        guard let raw = KeychainStore.value(for: set.storageKey),
              let data = raw.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(StoredToken.self, from: data)
    }

    private func save(_ token: StoredToken, for set: GoogleCredentialSet) {
        guard let data = try? JSONEncoder().encode(token) else { return }
        KeychainStore.setValue(String(data: data, encoding: .utf8), for: set.storageKey)
    }

    // ── PKCE ────────────────────────────────────────────────────────────

    private static func randomVerifier() -> String {
        var bytes = [UInt8](repeating: 0, count: 64)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return base64URL(Data(bytes))
    }

    private static func challenge(for verifier: String) -> String {
        base64URL(Data(SHA256.hash(data: Data(verifier.utf8))))
    }

    private static func base64URL(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    // ── presentation ────────────────────────────────────────────────────

    nonisolated func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        MainActor.assumeIsolated {
            let scene = UIApplication.shared.connectedScenes
                .compactMap { $0 as? UIWindowScene }
                .first { $0.activationState == .foregroundActive }
            return scene?.keyWindow ?? ASPresentationAnchor()
        }
    }
}
