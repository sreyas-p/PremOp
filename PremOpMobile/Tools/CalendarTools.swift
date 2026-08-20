import EventKit
import Foundation

/// Shared EventKit store. One instance: creating several is wasteful and each
/// carries its own permission state.
@MainActor
enum EventKitAccess {
    static let store = EKEventStore()

    static func requestCalendar() async -> Bool {
        (try? await store.requestFullAccessToEvents()) ?? false
    }

    static func requestReminders() async -> Bool {
        (try? await store.requestFullAccessToReminders()) ?? false
    }

    static func denied(_ what: String, toggle: String) -> String {
        """
        No permission to read \(what). iOS will not let this app proceed \
        without it. Ask the user to enable it in Settings › Privacy & Security \
        › \(toggle) › PremOpMobile. Do not retry until they confirm.
        """
    }
}

private let dateFormat: DateFormatter = {
    let f = DateFormatter()
    f.dateFormat = "EEE d MMM HH:mm"
    return f
}()

@MainActor
enum CalendarTools {
    static let all: [Tool] = [events, createEvent]

    static let events = Tool(
        name: "calendar_events",
        description: """
        List calendar events in a date range. Use this for anything about the \
        user's schedule — what they are doing, when they are free, whether \
        something is booked.

        Keep the window tight. Asking for a year returns hundreds of events and \
        buries the answer.
        """,
        schema: schema([
            "days_ahead": intProp("How many days forward to look. Use a negative number to look backwards, e.g. -7 for the past week. Defaults to 7."),
            "query": stringProp("Optional text to filter titles and locations by, case-insensitive.")
        ]),
        run: { input in
            guard await EventKitAccess.requestCalendar() else {
                return EventKitAccess.denied("the calendar", toggle: "Calendars")
            }
            let store = EventKitAccess.store
            let days = input["days_ahead"] as? Int ?? 7
            let now = Date()
            let other = Calendar.current.date(byAdding: .day, value: days, to: now) ?? now
            let (start, end) = days < 0 ? (other, now) : (now, other)

            let predicate = store.predicateForEvents(withStart: start, end: end, calendars: nil)
            var events = store.events(matching: predicate)

            if let query = (input["query"] as? String)?.lowercased(), !query.isEmpty {
                events = events.filter {
                    ($0.title ?? "").lowercased().contains(query)
                        || ($0.location ?? "").lowercased().contains(query)
                }
            }
            guard !events.isEmpty else {
                return "No events between \(dateFormat.string(from: start)) and \(dateFormat.string(from: end))."
            }

            let lines = events.prefix(60).map { event -> String in
                let when = event.isAllDay
                    ? "\(dateFormat.string(from: event.startDate)) (all day)"
                    : "\(dateFormat.string(from: event.startDate))–\(dateFormat.string(from: event.endDate))"
                let place = (event.location?.isEmpty == false) ? " @ \(event.location!)" : ""
                Indexer.remember(source: "calendar", id: event.eventIdentifier ?? UUID().uuidString,
                                 title: event.title ?? "", text: "\(event.title ?? "") \(when)\(place). \(event.notes ?? "")")
                return "- \(event.title ?? "(untitled)") — \(when)\(place) [\(event.calendar.title)]"
            }
            return "\(events.count) event(s):\n" + lines.joined(separator: "\n")
        }
    )

    static let createEvent = Tool(
        name: "calendar_create_event",
        description: """
        Create a calendar event. Only do this when the user has clearly asked \
        for something to be scheduled — it writes to their real calendar and \
        they will see it on every device.
        """,
        schema: schema([
            "title": stringProp("Event title."),
            "start": stringProp("Start time, ISO 8601, e.g. 2026-08-21T14:00:00."),
            "minutes": intProp("Duration in minutes. Defaults to 60."),
            "location": stringProp("Optional location."),
            "notes": stringProp("Optional notes.")
        ], required: ["title", "start"]),
        run: { input in
            guard await EventKitAccess.requestCalendar() else {
                return EventKitAccess.denied("the calendar", toggle: "Calendars")
            }
            guard let title = input["title"] as? String,
                  let startText = input["start"] as? String,
                  let start = FlexibleDate.parse(startText) else {
                return "Need a title and a start time in ISO 8601 form, e.g. 2026-08-21T14:00:00."
            }
            let store = EventKitAccess.store
            guard let calendar = store.defaultCalendarForNewEvents else {
                return "No writable calendar is available on this device."
            }

            let event = EKEvent(eventStore: store)
            event.title = title
            event.startDate = start
            event.endDate = start.addingTimeInterval(Double(input["minutes"] as? Int ?? 60) * 60)
            event.location = input["location"] as? String
            event.notes = input["notes"] as? String
            event.calendar = calendar

            do {
                try store.save(event, span: .thisEvent)
                return "Created '\(title)' on \(dateFormat.string(from: start)) in \(calendar.title)."
            } catch {
                return "Could not save the event: \(error.localizedDescription)"
            }
        }
    )
}

/// Parses the several date shapes the model actually emits.
///
/// Asking it for one exact format and rejecting the rest fails often enough to
/// be worth handling here instead: it produces ISO with and without fractional
/// seconds, bare local times with no zone, and plain dates.
enum FlexibleDate {
    private static let isoFractional: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private static let iso: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    private static let localFormats = [
        "yyyy-MM-dd'T'HH:mm:ss", "yyyy-MM-dd HH:mm", "yyyy-MM-dd"
    ]

    static func parse(_ string: String) -> Date? {
        if let date = isoFractional.date(from: string) { return date }
        if let date = iso.date(from: string) { return date }
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        for format in localFormats {
            formatter.dateFormat = format
            if let date = formatter.date(from: string) { return date }
        }
        return nil
    }
}
