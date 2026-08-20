import Foundation

/// What makes this a dispatch system rather than one large agent: a subagent
/// runs in its own conversation with only its own tools, and only its summary
/// comes back to the coordinator.
@MainActor
enum DelegateTools {
    static let all: [Tool] = [listAgents, delegate, delegateParallel]

    /// Depth cap, mirroring the desktop version. Worker agents hold no delegate
    /// tools, so this is defence in depth rather than the only guard.
    static let maxDepth = 2
    private static var depth = 0

    static let listAgents = Tool(
        name: "list_agents",
        description: "List the specialists available to delegate to and what each can reach.",
        schema: schema([:]),
        run: { _ in
            Agents.all.map { "- \($0.name): \($0.summary)\n  tools: \($0.toolNames.joined(separator: ", "))" }
                .joined(separator: "\n")
        }
    )

    static let delegate = Tool(
        name: "delegate_to_agent",
        description: """
        Run one specialist on a self-contained subtask and return its result.

        Use this only when a subtask depends on an earlier one's output. For \
        independent work use delegate_parallel, which runs them at once.

        The subagent sees only what you write here — restate the goal, the \
        concrete inputs, and the shape of the answer you want.
        """,
        schema: schema([
            "agent": stringProp("Which specialist to run."),
            "instructions": stringProp("A complete, self-contained brief.")
        ], required: ["agent", "instructions"]),
        run: { input in
            guard let name = input["agent"] as? String,
                  let brief = input["instructions"] as? String else {
                return "Both agent and instructions are required."
            }
            return await runSub(name: name, brief: brief)
        }
    )

    static let delegateParallel = Tool(
        name: "delegate_parallel",
        description: """
        Run several specialists at once and return all their results together.

        This is the default for a request with independent parts — "what's on \
        my calendar and how did I sleep" is two unrelated tasks, and running \
        them concurrently costs the time of the slower one rather than both.

        The lists are positional: agents[i] gets instructions[i], so they must \
        be the same length.
        """,
        schema: schema([
            "agents": ["type": "array", "items": ["type": "string"],
                       "description": "Specialist names, one per subtask."],
            "instructions": ["type": "array", "items": ["type": "string"],
                             "description": "Complete briefs, aligned with agents."]
        ], required: ["agents", "instructions"]),
        run: { input in
            guard let names = input["agents"] as? [String],
                  let briefs = input["instructions"] as? [String] else {
                return "Both agents and instructions must be arrays of strings."
            }
            guard names.count == briefs.count else {
                return "Refused: \(names.count) agents but \(briefs.count) instructions. They are positional and must match."
            }
            guard !names.isEmpty else { return "Refused: nothing to delegate." }
            guard depth < maxDepth else {
                return "Delegation refused: already \(depth) levels deep. Do the work directly."
            }

            let pairs = Array(zip(names, briefs)).enumerated()
            let results = await withTaskGroup(of: (Int, String).self) { group -> [(Int, String)] in
                for (index, pair) in pairs {
                    group.addTask { @MainActor in
                        (index, await runSub(name: pair.0, brief: pair.1))
                    }
                }
                var collected: [(Int, String)] = []
                for await item in group { collected.append(item) }
                return collected.sorted { $0.0 < $1.0 }
            }
            return "\(results.count) specialist(s) finished.\n\n"
                + results.map(\.1).joined(separator: "\n\n")
        }
    )

    private static func runSub(name: String, brief: String) async -> String {
        guard let spec = Agents.named(name) else {
            return "No agent named \(name). Available: \(Agents.all.map(\.name).joined(separator: ", "))"
        }
        guard depth < maxDepth else {
            return "Delegation refused: already \(depth) levels deep. Do the work directly."
        }
        depth += 1
        defer { depth -= 1 }

        let (tools, _) = ToolRegistry.resolve(spec.toolNames)
        do {
            let result = try await AnthropicClient().run(
                system: spec.system, prompt: brief, tools: tools, maxTokens: spec.maxTokens)
            return "### \(name)\n\(result.text)"
        } catch {
            return "### \(name)\nfailed: \(error.localizedDescription)"
        }
    }
}
