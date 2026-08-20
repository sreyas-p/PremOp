import Foundation

/// Minimal Messages API client with a tool-use loop.
///
/// Hand-rolled rather than pulled from a package: the whole surface needed here
/// is one endpoint and one loop, and the request/response shapes are stable.
actor AnthropicClient {
    struct Usage { var inputTokens = 0; var outputTokens = 0 }

    struct Result {
        var text: String
        var toolCalls: [String]
        var usage: Usage
    }

    enum ClientError: LocalizedError {
        case missingKey
        case http(Int, String)
        case malformed(String)

        var errorDescription: String? {
            switch self {
            case .missingKey:
                return "No Anthropic API key. Add one in Settings."
            case let .http(code, body):
                if code == 401 { return "Anthropic rejected the API key (401). Check it in Settings." }
                return "Anthropic returned \(code): \(body.prefix(200))"
            case let .malformed(detail):
                return "Unexpected response from Anthropic: \(detail)"
            }
        }
    }

    private let model: String
    private let session = URLSession(configuration: .default)

    init(model: String = "claude-opus-5") { self.model = model }

    /// Run one agent to completion: request, execute tools, repeat until the
    /// model stops asking for them.
    func run(
        system: String,
        prompt: String,
        tools: [Tool],
        maxTokens: Int,
        onProgress: @Sendable @escaping (String) -> Void = { _ in }
    ) async throws -> Result {
        guard let key = KeychainStore.apiKey, !key.isEmpty else { throw ClientError.missingKey }

        var messages: [[String: Any]] = [["role": "user", "content": prompt]]
        var calls: [String] = []
        var usage = Usage()
        let byName = Dictionary(uniqueKeysWithValues: tools.map { ($0.name, $0) })

        // A hard ceiling: without it a confused model can loop on tools until
        // the bill is the only thing that stops it.
        for _ in 0..<12 {
            let response = try await send(
                key: key, system: system, messages: messages,
                tools: tools, maxTokens: maxTokens
            )

            if let u = response["usage"] as? [String: Any] {
                usage.inputTokens += u["input_tokens"] as? Int ?? 0
                usage.outputTokens += u["output_tokens"] as? Int ?? 0
            }

            let content = response["content"] as? [[String: Any]] ?? []
            let stop = response["stop_reason"] as? String

            if stop != "tool_use" {
                return Result(text: Self.text(from: content), toolCalls: calls, usage: usage)
            }

            messages.append(["role": "assistant", "content": content])

            var results: [[String: Any]] = []
            for block in content where block["type"] as? String == "tool_use" {
                guard let name = block["name"] as? String,
                      let id = block["id"] as? String else { continue }
                calls.append(name)
                onProgress("· \(name)")

                let input = block["input"] as? [String: Any] ?? [:]
                let output: String
                if let tool = byName[name] {
                    output = await tool.run(input)
                } else {
                    output = "No such tool: \(name). Available: \(byName.keys.sorted().joined(separator: ", "))"
                }
                results.append([
                    "type": "tool_result", "tool_use_id": id, "content": output
                ])
            }

            if results.isEmpty {
                return Result(text: Self.text(from: content), toolCalls: calls, usage: usage)
            }
            messages.append(["role": "user", "content": results])
        }

        return Result(
            text: "Stopped after 12 tool rounds without a final answer.",
            toolCalls: calls, usage: usage
        )
    }

    private func send(
        key: String, system: String, messages: [[String: Any]],
        tools: [Tool], maxTokens: Int
    ) async throws -> [String: Any] {
        var request = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 180
        request.setValue(key, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var body: [String: Any] = [
            "model": model,
            "max_tokens": maxTokens,
            "system": system,
            "messages": messages
        ]
        if !tools.isEmpty { body["tools"] = tools.map(\.wireFormat) }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else {
            throw ClientError.http(code, String(data: data, encoding: .utf8) ?? "")
        }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ClientError.malformed("not a JSON object")
        }
        return object
    }

    private static func text(from content: [[String: Any]]) -> String {
        content
            .filter { $0["type"] as? String == "text" }
            .compactMap { $0["text"] as? String }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
