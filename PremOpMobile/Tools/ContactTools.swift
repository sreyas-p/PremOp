import Contacts
import Foundation

@MainActor
enum ContactTools {
    static let all: [Tool] = [search]

    static let search = Tool(
        name: "contacts_search",
        description: """
        Find people in the user's contacts by name, and return what is known \
        about them.

        Use this to resolve who someone is when a name appears in a calendar \
        event or a note and more detail would help. Match is on name only.
        """,
        schema: schema([
            "name": stringProp("Full or partial name. Matching is case-insensitive and partial."),
            "limit": intProp("Maximum people to return, 1-25. Defaults to 10.")
        ], required: ["name"]),
        run: { input in
            let store = CNContactStore()
            let granted = (try? await store.requestAccess(for: .contacts)) ?? false
            guard granted else {
                return EventKitAccess.denied("contacts", toggle: "Contacts")
            }
            guard let name = input["name"] as? String, !name.isEmpty else {
                return "A name to search for is required."
            }
            let limit = max(1, min(input["limit"] as? Int ?? 10, 25))

            let keys: [CNKeyDescriptor] = [
                CNContactGivenNameKey, CNContactFamilyNameKey,
                CNContactOrganizationNameKey, CNContactJobTitleKey,
                CNContactEmailAddressesKey, CNContactPhoneNumbersKey
            ].map { $0 as CNKeyDescriptor }

            do {
                let matches = try store.unifiedContacts(
                    matching: CNContact.predicateForContacts(matchingName: name), keysToFetch: keys)
                guard !matches.isEmpty else { return "No contact matches \(name)." }

                let lines = matches.prefix(limit).map { contact -> String in
                    var parts = ["\(contact.givenName) \(contact.familyName)"
                        .trimmingCharacters(in: .whitespaces)]
                    if !contact.organizationName.isEmpty {
                        let role = contact.jobTitle.isEmpty ? "" : "\(contact.jobTitle), "
                        parts.append("\(role)\(contact.organizationName)")
                    }
                    if let email = contact.emailAddresses.first?.value as String? {
                        parts.append(email)
                    }
                    if let phone = contact.phoneNumbers.first?.value.stringValue {
                        parts.append(phone)
                    }
                    return "- " + parts.joined(separator: " · ")
                }
                let note = matches.count > limit
                    ? "\n(\(matches.count) matched; showing \(limit).)" : ""
                return "\(matches.count) match(es):\n" + lines.joined(separator: "\n") + note
            } catch {
                return "Contact lookup failed: \(error.localizedDescription)"
            }
        }
    )
}
