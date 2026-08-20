import Foundation

/// One callable an agent can be granted.
///
/// `schema` is the JSON Schema handed to the model. The description is the
/// single biggest lever on whether a tool gets called correctly, so each one
/// says *when* to use it, not just what it does.
struct Tool {
    let name: String
    let description: String
    let schema: [String: Any]
    /// MainActor-isolated on purpose: every tool here touches EventKit, the
    /// Contacts store, or the index, all of which are main-actor state. Hopping
    /// once at the call site is simpler and safer than making each store
    /// Sendable to save a context switch that costs nothing next to a network
    /// round trip.
    let run: @MainActor ([String: Any]) async -> String

    var wireFormat: [String: Any] {
        ["name": name, "description": description, "input_schema": schema]
    }
}

/// Builds a JSON Schema object without the ceremony.
func schema(_ properties: [String: [String: Any]], required: [String] = []) -> [String: Any] {
    ["type": "object", "properties": properties, "required": required]
}

func stringProp(_ description: String) -> [String: Any] {
    ["type": "string", "description": description]
}

func intProp(_ description: String) -> [String: Any] {
    ["type": "integer", "description": description]
}

/// The registry. Agents name tools as strings; this resolves them.
@MainActor
enum ToolRegistry {
    private static var registry: [String: Tool] = {
        var all: [String: Tool] = [:]
        for tool in CalendarTools.all + ReminderTools.all + ContactTools.all
            + LifeTools.all + SearchTools.all + GoogleTools.all + DelegateTools.all {
            all[tool.name] = tool
        }
        return all
    }()

    static var names: [String] { registry.keys.sorted() }

    static func tool(named name: String) -> Tool? { registry[name] }

    /// Look up tools by name, preserving order. Unknown names are dropped and
    /// reported, rather than silently producing an under-equipped agent.
    static func resolve(_ names: [String]) -> (tools: [Tool], missing: [String]) {
        var found: [Tool] = []
        var missing: [String] = []
        for name in names {
            if let tool = registry[name] { found.append(tool) } else { missing.append(name) }
        }
        return (found, missing)
    }
}
