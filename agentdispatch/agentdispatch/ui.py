"""A local control panel for running and testing everything.

    python -m agentdispatch ui

Stdlib only — no framework, no build step, no new dependencies. Everything
slow (a dispatch, loading the 3B model, a sleep cycle) runs on a background
thread and the page polls, so the browser never hangs on a 90-second job.

This is a local tool by necessity, not preference: it needs your cached Google
tokens and the local model, so it cannot be a hosted page.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import agents
from .config import settings

# ── job plumbing ────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"state": "running", "log": [], "result": None, "error": None}
    return job_id


def _log(job_id: str, line: str) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            _jobs[job_id]["log"].append(f"{stamp}  {line}")


def _finish(job_id: str, result=None, error: str | None = None) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(
                state="failed" if error else "done", result=result, error=error
            )


def _run_async(job_id: str, fn) -> None:
    def wrapper():
        try:
            _finish(job_id, result=fn(lambda m: _log(job_id, m)))
        except Exception as exc:  # noqa: BLE001 — surfaced in the UI
            _finish(job_id, error=f"{type(exc).__name__}: {exc}")
            _log(job_id, traceback.format_exc(limit=4))

    threading.Thread(target=wrapper, daemon=True).start()


# ── the actual work ─────────────────────────────────────────────────────


def _job_dispatch(agent: str, prompt: str):
    def work(log):
        from .runner import run_agent, store

        log(f"dispatching to {agent!r} …")
        task = run_agent(agent, prompt)
        subtasks = store().list(parent_id=task.id, limit=50)
        log(f"finished: {task.status.value}")
        return {
            "task_id": task.id,
            "status": task.status.value,
            "text": task.result or task.error or "(no output)",
            "tools": task.tool_calls,
            "tokens_in": task.input_tokens,
            "tokens_out": task.output_tokens,
            "subtasks": [
                {"id": s.id, "agent": s.agent, "status": s.status.value}
                for s in subtasks
            ],
        }

    return work


def _job_memory(action: str, payload: dict):
    def work(log):
        from .tools.memory import _daemon

        log("loading local model (first call takes ~10s) …")
        daemon = _daemon()

        if action == "ask":
            log("querying weights …")
            return {"text": daemon.ask(payload["question"], actor="ui")}

        if action == "remember":
            log("solving MEMIT edit …")
            fact = daemon.remember(
                payload["subject"], payload["relation"], payload["target"],
                prompt=payload["prompt"], actor="ui",
                source=payload.get("source") or None,
            )
            return {"text": f"Learned: {fact.prompt} -> {fact.target}\n"
                            f"id {fact.id} | stage {int(fact.stage)}"}

        if action == "audit":
            report = daemon.audit()
            return {"text": json.dumps(report.model_dump(mode="json"), indent=2)}

        if action == "sleep":
            log("consolidating into LoRA — this takes a while …")
            report = daemon.sleep(actor="ui")
            return {"text": json.dumps(report.model_dump(mode="json"), indent=2)}

        raise ValueError(f"unknown memory action {action!r}")

    return work


def _status() -> dict:
    import os

    from .integrations.google_auth import CREDENTIAL_SETS

    try:
        import mlx.core  # noqa: F401

        mlx_ok = True
    except Exception:  # noqa: BLE001
        mlx_ok = False

    return {
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "model": settings.model,
        "mlx": mlx_ok,
        "google": [
            {"name": cs.name, "ok": cs.token_path.exists(),
             "scopes": [s.rsplit("/", 1)[-1] for s in cs.scopes]}
            for cs in CREDENTIAL_SETS
        ],
        "agents": [
            {"name": spec.name, "description": spec.description,
             "tools": spec.tools}
            for spec in agents.AGENTS.values()
        ],
    }


def _tasks(limit: int = 25) -> list[dict]:
    from .runner import store

    return [
        {
            "id": t.id, "agent": t.agent, "status": t.status.value,
            "created": t.created_at.strftime("%m-%d %H:%M"),
            "instructions": t.instructions,
            "text": t.result or t.error or "",
            "tools": t.tool_calls,
            "tokens": f"{t.input_tokens} in / {t.output_tokens} out",
        }
        for t in store().list(limit=limit)
    ]


# ── HTTP ────────────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:  # quiet
        pass

    def _send(self, payload, status: int = 200, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(PAGE.encode(), content_type="text/html; charset=utf-8")
        if path == "/api/status":
            return self._send(_status())
        if path == "/api/tasks":
            return self._send(_tasks())
        if path == "/api/job":
            job_id = self.path.split("id=")[-1]
            with _jobs_lock:
                job = _jobs.get(job_id)
            return self._send(job or {"state": "missing"}, 200 if job else 404)
        return self._send({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        path = self.path.split("?")[0]

        if path == "/api/run":
            job_id = _new_job()
            _run_async(job_id, _job_dispatch(payload["agent"], payload["prompt"]))
            return self._send({"job": job_id})

        if path == "/api/memory":
            job_id = _new_job()
            _run_async(job_id, _job_memory(payload.pop("action"), payload))
            return self._send({"job": job_id})

        return self._send({"error": "not found"}, 404)


def serve(port: int = 8765, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Control panel on {url}   (ctrl-c to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>PremOp</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#8b93a7;
--accent:#7aa2f7;--ok:#7bd88f;--bad:#f7768e;--warn:#e0af68}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.02em}
.pill{font-size:11px;padding:3px 9px;border-radius:99px;border:1px solid var(--line);
color:var(--dim)}
.pill.ok{color:var(--ok);border-color:#2c4534}.pill.bad{color:var(--bad);border-color:#4a2b33}
nav{display:flex;gap:2px;padding:0 20px;border-bottom:1px solid var(--line)}
nav button{background:none;border:0;color:var(--dim);padding:11px 15px;cursor:pointer;
font-size:13px;border-bottom:2px solid transparent}
nav button.on{color:var(--fg);border-bottom-color:var(--accent)}
main{padding:20px;max-width:1100px}
section{display:none}section.on{display:block}
label{display:block;font-size:12px;color:var(--dim);margin:12px 0 5px}
select,input,textarea{width:100%;background:var(--panel);color:var(--fg);
border:1px solid var(--line);border-radius:7px;padding:9px 11px;font:inherit}
textarea{min-height:88px;resize:vertical}
button.go{background:var(--accent);color:#0d1017;border:0;border-radius:7px;
padding:9px 18px;font-weight:600;cursor:pointer;margin-top:14px}
button.go:disabled{opacity:.5;cursor:default}
button.ghost{background:var(--panel);color:var(--fg);border:1px solid var(--line);
border-radius:7px;padding:8px 14px;cursor:pointer;margin:14px 8px 0 0}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:13px;white-space:pre-wrap;word-break:break-word;margin-top:16px;
max-height:460px;overflow:auto;font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}
.meta{color:var(--dim);font-size:12px;margin-top:9px}
.row{display:flex;gap:14px;flex-wrap:wrap}.row>div{flex:1;min-width:180px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:12px 14px;margin-bottom:9px;cursor:pointer}
.card h4{margin:0 0 4px;font-size:13px;font-weight:600}
.card .s{font-size:11px;color:var(--dim)}
.ex{color:var(--accent);cursor:pointer;font-size:12px;display:block;margin:5px 0}
</style></head><body>
<header><h1>PremOp</h1><span id="badges"></span></header>
<nav>
 <button class="on" data-t="run">Run</button>
 <button data-t="memory">Memory</button>
 <button data-t="tasks">History</button>
 <button data-t="health">Health</button>
</nav>
<main>

<section id="run" class="on">
  <label>Agent</label><select id="agent"></select>
  <div class="meta" id="agentinfo"></div>
  <label>Prompt</label><textarea id="prompt" placeholder="What should it do?"></textarea>
  <div id="examples"></div>
  <button class="go" id="rungo">Run</button>
  <pre id="runout">Ready.</pre>
</section>

<section id="memory">
  <div class="row">
    <div><label>Ask the local model</label>
      <input id="mq" placeholder="Zilbex Corp is headquartered in">
      <div class="meta">Phrase as a sentence opening, not a question.</div>
      <button class="go" id="askgo">Ask</button></div>
  </div>
  <label style="margin-top:22px">Teach it a fact</label>
  <div class="row">
    <div><input id="msub" placeholder="subject"></div>
    <div><input id="mrel" placeholder="relation"></div>
    <div><input id="mtar" placeholder="target"></div>
  </div>
  <input id="mpr" placeholder="cloze prompt, e.g. 'Zilbex Corp is headquartered in'" style="margin-top:9px">
  <button class="go" id="remgo">Remember</button>
  <button class="ghost" id="auditgo">Audit</button>
  <button class="ghost" id="sleepgo">Sleep (consolidate)</button>
  <pre id="memout">Ready. First call loads the 3B model (~10s).</pre>
</section>

<section id="tasks">
  <button class="ghost" id="refresh">Refresh</button>
  <div id="tasklist"></div>
  <pre id="taskout">Click a run to see it in full.</pre>
</section>

<section id="health"><pre id="healthout">loading…</pre></section>
</main>
<script>
const $=s=>document.querySelector(s), api=(u,o)=>fetch(u,o).then(r=>r.json());
let STATUS=null;

document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('on',x===b));
  document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id===b.dataset.t));
  if(b.dataset.t==='tasks') loadTasks();
});

const EX={
 mail:["How many unread emails do I have from the last 7 days? Just the count.",
       "Find receipts from this month and list vendor, amount, and date.",
       "Find my flight confirmations, remember airline and date for each."],
 youtube:["Summarize the last 5 videos I liked, grouped by topic.",
          "List my subscriptions and what each channel covers."],
 notetaker:["Write a note titled 'Test' with three bullet points about anything."],
 dispatcher:["Check my recent mail and my liked videos, and tell me if any topic appears in both."]};

function paint(){
  $('#badges').innerHTML =
    `<span class="pill ${STATUS.anthropic?'ok':'bad'}">anthropic ${STATUS.anthropic?'ok':'missing'}</span> `
    + STATUS.google.map(g=>`<span class="pill ${g.ok?'ok':'bad'}">${g.name} ${g.ok?'ok':'no consent'}</span>`).join(' ')
    + ` <span class="pill ${STATUS.mlx?'ok':'bad'}">mlx ${STATUS.mlx?'ok':'absent'}</span>`
    + ` <span class="pill">${STATUS.model}</span>`;
  $('#agent').innerHTML = STATUS.agents.map(a=>`<option value="${a.name}">${a.name}</option>`).join('');
  showAgent();
  $('#healthout').textContent = JSON.stringify(STATUS,null,2);
}
function showAgent(){
  const a = STATUS.agents.find(x=>x.name===$('#agent').value);
  $('#agentinfo').textContent = a.description+'  ·  tools: '+a.tools.join(', ');
  $('#examples').innerHTML = (EX[a.name]||[]).map(e=>`<span class="ex">▸ ${e}</span>`).join('');
  document.querySelectorAll('#examples .ex').forEach(el=>el.onclick=()=>$('#prompt').value=el.textContent.slice(2).trim());
}
$('#agent').onchange=showAgent;

async function poll(job,out,btn){
  const t=setInterval(async()=>{
    const j=await api('/api/job?id='+job);
    out.textContent=(j.log||[]).join('\n')+(j.state==='running'?'\n…':'');
    if(j.state==='running')return;
    clearInterval(t); if(btn)btn.disabled=false;
    if(j.error){out.textContent+='\n\nFAILED\n'+j.error;return;}
    const r=j.result||{};
    let s=(j.log||[]).join('\n')+'\n\n'+(r.text||'');
    if(r.tools&&r.tools.length)s+='\n\ntools: '+r.tools.join(', ');
    if(r.tokens_in!=null)s+=`\ntokens: ${r.tokens_in} in / ${r.tokens_out} out  ·  ${r.task_id}`;
    if(r.subtasks&&r.subtasks.length)s+='\nsubtasks: '+r.subtasks.map(x=>`${x.agent}(${x.status})`).join(', ');
    out.textContent=s;
  },700);
}
$('#rungo').onclick=async()=>{
  const p=$('#prompt').value.trim(); if(!p)return;
  $('#rungo').disabled=true; $('#runout').textContent='starting…';
  const {job}=await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({agent:$('#agent').value,prompt:p})});
  poll(job,$('#runout'),$('#rungo'));
};
function mem(body,btn){
  return async()=>{
    btn.disabled=true; $('#memout').textContent='starting…';
    const {job}=await api('/api/memory',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body())});
    poll(job,$('#memout'),btn);
  };
}
$('#askgo').onclick=mem(()=>({action:'ask',question:$('#mq').value}),$('#askgo'));
$('#remgo').onclick=mem(()=>({action:'remember',subject:$('#msub').value,relation:$('#mrel').value,
  target:$('#mtar').value,prompt:$('#mpr').value,source:'ui'}),$('#remgo'));
$('#auditgo').onclick=mem(()=>({action:'audit'}),$('#auditgo'));
$('#sleepgo').onclick=mem(()=>({action:'sleep'}),$('#sleepgo'));

async function loadTasks(){
  const ts=await api('/api/tasks');
  $('#tasklist').innerHTML = ts.length?ts.map((t,i)=>
    `<div class="card" data-i="${i}"><h4>${t.agent} · ${t.status}</h4>
     <div class="s">${t.created} · ${t.tokens}</div>
     <div class="s">${t.instructions.slice(0,110)}</div></div>`).join(''):'<p class="meta">No runs yet.</p>';
  document.querySelectorAll('#tasklist .card').forEach(c=>c.onclick=()=>{
    const t=ts[c.dataset.i];
    $('#taskout').textContent=`${t.id}  ${t.agent}  ${t.status}  ${t.tokens}
tools: ${t.tools.join(', ')||'—'}

PROMPT
${t.instructions}

OUTPUT
${t.text}`;});
}
$('#refresh').onclick=loadTasks;
api('/api/status').then(s=>{STATUS=s;paint();});
</script></body></html>
"""
