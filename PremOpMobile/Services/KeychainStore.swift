import Foundation
import Security

/// Keychain-backed secret storage.
///
/// Secrets are still on the device, which is fine for a personal build and
/// wrong for anything distributed — a determined owner of the device can reach
/// them. A distributed version needs a relay holding credentials server-side.
enum KeychainStore {
    private static let service = "com.sreyasprabu.PremOpMobile"

    static func value(for account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: account,
            kSecAttrService as String: service,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func setValue(_ newValue: String?, for account: String) {
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: account,
            kSecAttrService as String: service
        ]
        SecItemDelete(base as CFDictionary)
        guard let value = newValue?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty, let data = value.data(using: .utf8) else { return }

        var insert = base
        insert[kSecValueData as String] = data
        insert[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(insert as CFDictionary, nil)
    }

    // ── named secrets ───────────────────────────────────────────────────

    static var apiKey: String? {
        get { value(for: "anthropic-api-key") }
        set { setValue(newValue, for: "anthropic-api-key") }
    }

    /// The iOS OAuth client ID. Public by design — iOS clients have no secret —
    /// but kept here rather than in UserDefaults so everything sensitive lives
    /// in one place.
    static var googleClientID: String? {
        get { value(for: "google-client-id") }
        set { setValue(newValue, for: "google-client-id") }
    }

    static var hasKey: Bool { !(apiKey ?? "").isEmpty }
}
