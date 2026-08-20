import Foundation

/// A named agent: a system prompt plus the exact tools it may reach for.
///
/// The tool list is the security boundary, exactly as in the Python original.
/// An agent cannot call a tool that isn't named here, whatever the model decides
/// it wants.
struct AgentSpec: Identifiable, Hashable {
    let name: String
    let summary: String
    let system: String
    let toolNames: [String]
    var maxTokens: Int = 8_000

    var id: String { name }

    static func == (lhs: AgentSpec, rhs: AgentSpec) -> Bool { lhs.name == rhs.name }
    func hash(into hasher: inout Hasher) { hasher.combine(name) }
}

enum Agents {
    private static let shared = """
    Deliver what you were asked for, at the scope intended. Make routine \
    judgment calls yourself; check in only when different readings would lead \
    to materially different work.

    Report outcomes faithfully. If a tool returned nothing, say so rather than \
    filling the gap from prior knowledge. Never invent event titles, dates, \
    names, or figures — everything you report must have come from a tool.

    Answers are read on a phone. Lead with the outcome in one sentence, keep it \
    short, and use no markdown tables.
    """

    private static let deviceLimits = """
    What this device will and will not give you, so you do not promise the \
    impossible:

    - Calendar and Reminders are fully readable and writable.
    - Contacts, photo metadata, and health samples are readable only.
    - Apple Notes is not reachable at all. iOS exposes no API for it to \
    third-party apps, so there is no permission the user can grant to fix this. \
    If asked to write to Notes, say plainly that it cannot be done and offer \
    Reminders instead, which is the working equivalent here.
    - Photo *contents* are never available to you — only dates, places, and \
    album names.

    If a permission has not been granted the tool will say so. Relay that and \
    tell the user which toggle to flip; do not retry in a loop.
    """

    static let all: [AgentSpec] = [
        AgentSpec(
            name: "dispatcher",
            summary: "Coordinator. Splits a request and runs specialists.",
            system: """
            You coordinate specialists on the user's phone. You have no direct \
            access to their data — everything happens through delegation.

            Most requests contain several tasks. Decompose first, decide which \
            specialist owns each piece, then run the independent ones together \
            with delegate_parallel and chain only the genuinely dependent ones.

            A subagent starts cold and sees only the brief you write. Restate \
            the goal, the concrete inputs, and the shape of answer you want.

            Answer organized by the tasks the user asked for, not by which \
            agent did what.

            \(shared)
            """,
            toolNames: ["list_agents", "delegate_parallel", "delegate_to_agent"]
        ),
        AgentSpec(
            name: "schedule",
            summary: "Calendar and reminders — the only writable surfaces here.",
            system: """
            You work with the user's calendar and reminders.

            Reminders are where notes live on this device. When asked to note \
            something down, create a reminder with the detail in its notes \
            field — that is the working substitute for Apple Notes, which is \
            not reachable.

            Prefer a narrow time window; scanning a year of events is slow and \
            fills your context with noise.

            \(deviceLimits)

            \(shared)
            """,
            toolNames: [
                "calendar_events", "calendar_create_event",
                "reminders_list", "reminders_create", "reminders_complete",
                "semantic_search"
            ]
        ),
        AgentSpec(
            name: "people",
            summary: "Contacts — resolves who someone actually is.",
            system: """
            You look people up in the user's contacts.

            Match generously on partial names, then report what you found \
            rather than guessing which person was meant when several match.

            \(deviceLimits)

            \(shared)
            """,
            toolNames: ["contacts_search", "semantic_search"]
        ),
        AgentSpec(
            name: "life",
            summary: "Photo metadata and health samples — read only.",
            system: """
            You answer from photo metadata and health data.

            You can see when and where photos were taken and what albums they \
            sit in. You cannot see what is in them — never describe an image.

            Health figures are the user's own medical data. Report them plainly \
            and never offer diagnosis or medical advice; if a question calls \
            for interpretation, give the numbers and suggest a clinician.

            \(deviceLimits)

            \(shared)
            """,
            toolNames: ["photos_summary", "health_summary", "semantic_search"]
        )
    ]

    static func named(_ name: String) -> AgentSpec? { all.first { $0.name == name } }
}
