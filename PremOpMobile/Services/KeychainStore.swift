import Foundation
import Security

/// The Anthropic key lives in the keychain, never in the bundle or UserDefaults.
///
/// It is still on the device, which is fine for a personal build and is not
/// fine for anything distributed — a determined owner of the device can reach
/// it. A distributed version needs a relay that holds the key server-side.
enum KeychainStore {
    private static let account = "anthropic-api-key"
    private static let service = "com.sreyasprabu.PremOpMobile"

    static var apiKey: String? {
        get {
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
        set {
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
    }

    static var hasKey: Bool { !(apiKey ?? "").isEmpty }
}
