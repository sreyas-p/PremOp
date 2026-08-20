# PremOpMobile

PremOp as a native iOS app: Claude agents that read your on-device data,
delegate to each other, and take notes where iOS actually allows it.

An iOS fork of [PremOp](https://github.com/sreyas-p/PremOp) — same dispatch
architecture, rewritten in Swift against what a sandboxed iPhone app can
genuinely reach.

## What it can reach, and what it can't

| | |
| --- | --- |
| **Calendar** | read **and write** (EventKit) |
| **Reminders** | read **and write** — this is where notes go |
| **Contacts** | read only |
| **Photos** | metadata only — dates, places, albums. Never image contents. |
| **Health** | read only — steps, sleep, workouts |
| **Apple Notes** | **impossible.** iOS exposes no API to third-party apps. |
| **Mail / Messages** | **impossible.** No read API exists. |

Apple Notes is the one people expect and it genuinely cannot be done — it is
not a permission the user can grant. **Reminders are the working substitute**:
a reminder with its notes field filled in syncs across devices, is searchable
in a first-party app, and survives this app being deleted. Every agent prompt
states this so the model offers Reminders instead of promising Notes.

The documented workaround, if you need existing Apple Notes content: export
them to Files, then read them through the document picker.

## Agents

| | |
| --- | --- |
| `dispatcher` | Coordinator. Splits a request, fans out independent parts. |
| `schedule` | Calendar and reminders — the only writable surfaces. |
| `people` | Contacts lookup. |
| `life` | Photo metadata and health samples. |

Multi-task prompts work the same way as on desktop: the dispatcher decomposes
the request and runs independent parts concurrently via `delegate_parallel`.
Worker agents hold no delegation tools, and depth is capped at 2.

## Semantic search, free

Embeddings come from `NLEmbedding` in iOS's Natural Language framework — no
model download, no MLX, no network. That is a better trade on a phone than the
130MB embedder the desktop version uses: it is already installed, it is fast,
and it costs nothing in app size. The index fills as agents read your calendar
and reminders, and never leaves the device.

## Running it

```bash
open PremOpMobile.xcodeproj
```

Pick a simulator and hit ⌘R. Paste an Anthropic key into Settings on first
launch — it goes to the iOS keychain, never the bundle.

For a physical device you also need a signing team: select the target →
Signing & Capabilities → check *Automatically manage signing*. Change the
bundle identifier if `com.sreyasprabu.PremOpMobile` is taken.

**The key sits on the device.** That is fine for a personal build and wrong for
anything distributed — a distributed version needs a relay holding the key
server-side.

## Not built yet

- **Weight-based memory.** `memorydaemon` has no iOS port. A 3B bf16 model is
  6.4GB against a ~6GB per-app ceiling, and 4-bit weights cannot be MEMIT-edited
  — so on-device weight memory means dropping to a 1B bf16 (~2.5GB). The
  semantic index covers recall for now.
- **Files / document picker**, the Apple Notes export path.
- **Gmail, YouTube, Google Docs.** The desktop agents' Google integrations are
  not ported; this version reads the phone, not your Google account.
