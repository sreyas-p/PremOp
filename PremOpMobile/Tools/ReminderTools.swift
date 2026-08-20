import EventKit
import Foundation

/// Reminders are where notes live on this device.
///
/// Apple Notes exposes no API to third-party apps, so a reminder with its notes
/// field filled in is the closest working equivalent: it syncs across the
/// user's devices, it is searchable in a first-party app, and unlike a note in
/// our own sandbox it survives this app being deleted.
@MainActor
enum ReminderTools {
    static let all: [Tool] = [list, create, complete]

    private static func fetch(_ predicate: NSPredicate) async -> [EKReminder] {
        await withCheckedContinuation { continuation in
            EventKitAccess.store.fetchReminders(matching: predicate) { reminders in
                continuation.resume(returning: reminders ?? [])
            }
        }
    }

    static let list = Tool(
        name: "reminders_list",
        description: """
        List reminders, which on this device double as the user's notes.

        Search here before creating a new reminder on a recurring subject, so \
        entries extend an existing one instead of scattering across duplicates.
        """,
        schema: schema([
            "include_completed": ["type": "boolean", "description": "Include finished reminders. Defaults to false."],
            "query": stringProp("Optional text to filter titles and notes by, case-insensitive.")
        ]),
        run: { input in
            guard await EventKitAccess.requestReminders() else {
                return EventKitAccess.denied("reminders", toggle: "Reminders")
            }
            let store = EventKitAccess.store
            let includeCompleted = input["include_completed"] as? Bool ?? false
            let predicate = includeCompleted
                ? store.predicateForReminders(in: nil)
                : store.predicateForIncompleteReminders(
                    withDueDateStarting: nil, ending: nil, calendars: nil)

            var reminders = await fetch(predicate)
            if let query = (input["query"] as? String)?.lowercased(), !query.isEmpty {
                reminders = reminders.filter {
                    $0.title.lowercased().contains(query)
                        || ($0.notes ?? "").lowercased().contains(query)
                }
            }
            guard !reminders.isEmpty else { return "No matching reminders." }

            let lines = reminders.prefix(60).map { reminder -> String in
                Indexer.remember(
                    source: "reminder",
                    id: reminder.calendarItemIdentifier,
                    title: reminder.title,
                    text: "\(reminder.title). \(reminder.notes ?? "")"
                )
                var line = "- \(reminder.title)"
                if let due = reminder.dueDateComponents?.date {
                    line += " (due \(dateFormatShort.string(from: due)))"
                }
                if reminder.isCompleted { line += " [done]" }
                if let notes = reminder.notes, !notes.isEmpty {
                    line += "\n    \(notes.prefix(300))"
                }
                return line + "\n    id=\(reminder.calendarItemIdentifier)"
            }
            return "\(reminders.count) reminder(s):\n" + lines.joined(separator: "\n")
        }
    )

    static let create = Tool(
        name: "reminders_create",
        description: """
        Create a reminder. This is how you take a note on this device — put the \
        substance in `notes`, not just a bare title.

        Write for someone who did not see your work: give it a specific title, \
        attribute claims to where they came from, and include dates and names \
        needed to make sense of it later.
        """,
        schema: schema([
            "title": stringProp("Short, specific title. 'Rent increase Oct 2026', not 'Note'."),
            "notes": stringProp("The body of the note. This is the important field."),
            "due": stringProp("Optional due date, ISO 8601. Only set when the user asked for a reminder in time, not for plain notes.")
        ], required: ["title"]),
        run: { input in
            guard await EventKitAccess.requestReminders() else {
                return EventKitAccess.denied("reminders", toggle: "Reminders")
            }
            guard let title = input["title"] as? String else { return "A title is required." }
            let store = EventKitAccess.store
            guard let calendar = store.defaultCalendarForNewReminders() else {
                return "No writable reminders list is available on this device."
            }

            let reminder = EKReminder(eventStore: store)
            reminder.title = title
            reminder.notes = input["notes"] as? String
            reminder.calendar = calendar
            if let dueText = input["due"] as? String, let due = FlexibleDate.parse(dueText) {
                reminder.dueDateComponents = Calendar.current.dateComponents(
                    [.year, .month, .day, .hour, .minute], from: due)
            }

            do {
                try store.save(reminder, commit: true)
                Indexer.remember(source: "reminder", id: reminder.calendarItemIdentifier,
                                 title: title, text: "\(title). \(input["notes"] as? String ?? "")")
                return "Saved '\(title)' to \(calendar.title).\nid=\(reminder.calendarItemIdentifier)"
            } catch {
                return "Could not save the reminder: \(error.localizedDescription)"
            }
        }
    )

    static let complete = Tool(
        name: "reminders_complete",
        description: "Mark a reminder finished. Only when the user explicitly asks.",
        schema: schema(["id": stringProp("The id shown by reminders_list.")], required: ["id"]),
        run: { input in
            guard await EventKitAccess.requestReminders() else {
                return EventKitAccess.denied("reminders", toggle: "Reminders")
            }
            guard let id = input["id"] as? String,
                  let reminder = EventKitAccess.store.calendarItem(withIdentifier: id) as? EKReminder else {
                return "No reminder with that id. Call reminders_list to get current ids."
            }
            reminder.isCompleted = true
            do {
                try EventKitAccess.store.save(reminder, commit: true)
                return "Completed '\(reminder.title)'."
            } catch {
                return "Could not update it: \(error.localizedDescription)"
            }
        }
    )
}

let dateFormatShort: DateFormatter = {
    let f = DateFormatter()
    f.dateFormat = "d MMM"
    return f
}()
