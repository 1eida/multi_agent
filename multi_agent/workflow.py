"""Synchronous, task-local four-role evidence-driven workflow."""
from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from .common import Model, ModelBudgetError, Result, Task, Toolbox

from .agents import Planner, PlanValidator, Worker, WorkerValidator, empty_memory, normalize_memory
from .ledger import EvidenceLedger


@dataclass
class Config:
    max_rounds: int = 3
    min_rounds: int = 1
    allow_early_stop: bool = True
    max_actions_per_round: int = 6
    max_tool_calls: int = 16
    max_model_calls: int = 40
    max_seconds: float = 300.0
    enable_coevolution: bool = True
    evolve_roles: tuple[str, ...] = ("planner", "plan_validator", "worker", "worker_validator")
    evidence_gated_update: bool = True
    max_obs_chars: int = 4000
    max_total_evidence_chars: int = 24000
    method_name: str = "four_agent_coevolve"

    def __post_init__(self):
        if (self.max_rounds < 1 or not 1 <= self.min_rounds <= self.max_rounds
                or self.max_actions_per_round < 1 or self.max_tool_calls < 0
                or self.max_model_calls < 0 or self.max_seconds <= 0
                or self.max_obs_chars < 128 or self.max_total_evidence_chars < 128):
            raise ValueError("Invalid workflow budget or context limit")
        if set(self.evolve_roles) - {"planner", "plan_validator", "worker", "worker_validator"}:
            raise ValueError("Unknown role in evolve_roles")


class _Budget(Exception):
    pass


def _bounded(value: Any, limit: int) -> Any:
    """Bound untrusted non-ledger inputs while retaining JSON-compatible structure."""
    raw = json.dumps(value, ensure_ascii=False, default=str)
    if len(raw) <= limit:
        return copy.deepcopy(value)
    return {"truncated": True, "original_chars": len(raw), "preview": raw[:max(0, limit - 100)]}


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=True, default=str))


def _short_text(text: str, limit: int) -> str:
    """Keep both the opening question and any contract appended at its end."""
    if _json_size(text) <= limit:
        return text
    marker = f"\n[truncated; original_chars={len(text)}]\n"
    low, high = 0, len(text)
    while low < high:
        keep = (low + high + 1) // 2
        candidate = text[:(keep + 1) // 2] + marker + (text[-(keep // 2):] if keep // 2 else "")
        if _json_size(candidate) <= limit:
            low = keep
        else:
            high = keep - 1
    return text[:(low + 1) // 2] + marker + (text[-(low // 2):] if low // 2 else "")


def _structured_bound(value: Any, limit: int) -> Any:
    """Retain task dictionaries/lists and contract fields; shorten long text only."""
    body = copy.deepcopy(value)
    protected = {"task_id", "response_format", "output_contract", "table", "name"}
    if isinstance(body, str):
        return _short_text(body, limit)
    while _json_size(body) > limit:
        leaves = []

        def collect(node):
            if isinstance(node, dict):
                for key, item in node.items():
                    if key not in protected:
                        if isinstance(item, str) and _json_size(item) > 160:
                            leaves.append((node, key, item))
                        else:
                            collect(item)
            elif isinstance(node, list):
                for index, item in enumerate(node):
                    if isinstance(item, str) and _json_size(item) > 160:
                        leaves.append((node, index, item))
                    else:
                        collect(item)

        collect(body)
        if not leaves:
            raise ValueError("structured task metadata exceeds context budget; cannot remove required fields")
        parent, key, text = max(leaves, key=lambda item: _json_size(item[2]))
        target = max(160, _json_size(text) - (_json_size(body) - limit))
        parent[key] = _short_text(text, target)
    return body


def _task_view(task_data: dict) -> dict:
    return {"task_id": copy.deepcopy(task_data.get("task_id")),
            "question": _structured_bound(task_data.get("question", ""), 8000),
            "context": _structured_bound(task_data.get("context"), 16000)}


def _catalog_view(catalog: list[dict]) -> list[dict]:
    """Keep every tool and executable schema; compact descriptive prose only."""
    view = copy.deepcopy(catalog)
    if _json_size(view) > 12000:
        for tool in view:
            if isinstance(tool.get("description"), str):
                tool["description"] = _short_text(tool["description"], 512)
    return view


def _ids(value: Any) -> list[str]:
    found = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k in ("evidence_ids", "cited_evidence_ids"):
                found.extend([v] if isinstance(v, str) else v if isinstance(v, list) else [])
            elif k == "item_evidence_map" and isinstance(v, dict):
                for ids in v.values():
                    found.extend([ids] if isinstance(ids, str) else ids if isinstance(ids, list) else _ids(ids))
            else:
                found.extend(_ids(v))
    elif isinstance(value, list):
        for v in value:
            found.extend(_ids(v))
    return [v for v in found if isinstance(v, str)]


def _inspection_tools(catalog: list[dict]) -> set[str]:
    """Recognize detail capability from public tool documentation, not task names."""
    return {tool["name"] for tool in catalog
            if re.search(r"\b(full|complete|detail(?:ed|s)?)\b", tool.get("description", ""), re.I)
            and re.search(r"\b(text|document|abstract|record|content|detail(?:s)?)\b",
                          tool.get("description", ""), re.I)}


_MUTATING_WORDS = {"post", "create", "write", "update", "delete", "remove", "submit",
                   "send", "patch", "put", "insert", "upsert", "apply", "mutate",
                   "mutates", "mutation"}


def _words(value: Any) -> set[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    return {part.lower() for part in re.split(r"[^A-Za-z0-9]+", text) if part}


def _tool_policy(tool: dict) -> str:
    """Classify catalog capability conservatively; unknown tools are never mandatory."""
    if tool.get("mutates") is True or tool.get("read_only") is False:
        return "mutating"
    if (_words(tool.get("name", "")) | _words(tool.get("description", ""))) & _MUTATING_WORDS:
        return "mutating"
    if tool.get("read_only") is True or tool.get("mutates") is False:
        return "read_only"
    method = str(tool.get("http_method", tool.get("method", ""))).upper()
    return "read_only" if method in {"GET", "HEAD", "OPTIONS"} else "unknown"


def _tool_is_read_only(tool: dict) -> bool:
    return _tool_policy(tool) == "read_only"


def _terminal_failure(entry: dict) -> str | None:
    if entry.get("ok") is True:
        return None
    error = entry.get("error")
    if isinstance(error, dict):
        status = error.get("status_code", error.get("status"))
        try:
            code = int(status)
        except (TypeError, ValueError):
            code = None
        if code == 404 or error.get("not_found") is True:
            return "terminal_not_found"
        if code in {400, 408, 410, 429} or code is not None and 500 <= code <= 599:
            return "terminal_error"
        if code is not None:
            return None
        error = error.get("message", "")
    text = str(error or "").lower()
    if re.search(r"\b(?:http(?:error)?\s*)?404\b|\bnot[ -]?found\b", text):
        return "terminal_not_found"
    if re.search(r"\b(?:http(?:error)?\s*)?(?:400|408|410|429|5\d\d)\b", text):
        return "terminal_error"
    return None


def _terminal_read_failure(entry: dict) -> bool:
    return _terminal_failure(entry) is not None


def _covered_read_only(name: str, attempts_by_tool: dict[str, list[dict]]) -> bool:
    entries = attempts_by_tool.get(name, [])
    if any(e.get("ok") is True for e in entries):
        return True
    terminal = [e for e in entries if e.get("terminal_failure") or _terminal_read_failure(e)]
    return len(terminal) >= 2 and len({e.get("call_key") for e in terminal}) >= 2


def _has_previews(value: Any) -> bool:
    if isinstance(value, dict):
        return any((key in ("snippet", "preview") and bool(item)) or _has_previews(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_has_previews(item) for item in value)
    return False


def _format_only(review: dict | None) -> bool:
    """Exempt only an explicitly evidence-complete formatting correction."""
    return bool(review and review.get("evidence_complete") is True
                and review.get("format_fix_instruction")
                and not review.get("unresolved_issues"))


def _snap_changed(before: Any, after: Any) -> bool:
    """Compare opaque toolbox snapshots conservatively."""
    try:
        return before != after
    except Exception:
        return True


def run(task: Task, model: Model, tools: Toolbox, *, config: Config | None = None) -> Result:
    cfg = config or Config()
    started = time.monotonic()
    ledger = EvidenceLedger()
    trace: list[dict] = []
    roles = (Planner, PlanValidator, Worker, WorkerValidator)
    memories = {r.name: empty_memory(r.name) for r in roles}
    history = []
    calls = tool_calls = rounds = 0
    early = False
    best = None
    best_snapshot = None
    best_ids: set[str] = set()
    best_grounded = None
    best_grounded_snapshot = None
    mutated_call_keys: set[tuple[str, str]] = set()
    attempts_by_tool: dict[str, list[dict]] = {}
    previous = None
    previous_vp = None
    previous_vw = None
    status = "budget_exhausted"
    error = None
    catalog = tools.catalog()
    snapshot_fn = getattr(tools, "snapshot", None)
    restore_fn = getattr(tools, "restore", None)

    def snapshot_state():
        if not callable(snapshot_fn):
            return None
        return copy.deepcopy(snapshot_fn())

    def restore_state(snapshot):
        if callable(restore_fn):
            restore_fn(copy.deepcopy(snapshot))

    initial_tool_snapshot = snapshot_state()
    tool_info = {x["name"]: x for x in catalog}
    tool_policy = {name: _tool_policy(info) for name, info in tool_info.items()}
    inspection_tools = _inspection_tools(catalog)
    task_data = asdict(task) if is_dataclass(task) else {"task_id": getattr(task, "task_id", ""), "question": getattr(task, "question", str(task)), "context": getattr(task, "context", None)}

    finalization_seconds = min(5.0, cfg.max_seconds * 0.05)

    def check(*, reserved=False):
        limit = cfg.max_seconds if reserved else max(0.0, cfg.max_seconds - finalization_seconds)
        if time.monotonic() - started >= limit:
            raise _Budget("time")

    def ask(role, payload, *, reserved=False):
        nonlocal calls
        def invoke(body, repair=False):
            nonlocal calls
            check(reserved=reserved)
            reserve_final = cfg.max_model_calls >= 4
            limit = cfg.max_model_calls if reserved or not reserve_final else cfg.max_model_calls - 1
            if calls >= limit:
                raise _Budget("model calls")
            calls += 1
            trace.append({"event": "model_call", "role": role.name, "repair": repair, "round": current_round})
            structured = getattr(model, "ask_structured", None)
            if callable(structured):
                schema = copy.deepcopy(role.response_schema)
                if role is Worker:
                    if payload.get("tools_disabled"):
                        schema = Worker.constrained_schema(catalog, tools_disabled=True)
                    elif payload.get("required_tool_names"):
                        schema = Worker.constrained_schema(catalog,
                            required_tool_names=payload.get("required_tool_names"),
                            allow_final=payload.get("allow_final", True))
                    elif payload.get("force_empirical_tool"):
                        schema = Worker.constrained_schema(catalog, allow_final=False)
                    elif payload.get("force_inspection_tool"):
                        schema = Worker.constrained_schema(catalog, required_tool_names=payload.get("inspection_tool_names"), allow_final=False)
                    else:
                        schema = Worker.constrained_schema(catalog, allow_final=True)
                return structured(role.prompt, body, role=role.name, response_schema=schema)
            return model.ask(role.prompt, body, role=role.name, json_mode=True)
        body = {k: (_task_view(v) if k == "task" else
                    _catalog_view(v) if k == "catalog" else _bounded(v, 2400))
                for k, v in payload.items() if k != "evidence"}
        if "evidence" in payload:
            body["evidence"] = payload["evidence"]
        def fit(body):
            while _json_size(body) > 48000:
                evidence = body.get("evidence", "")
                if isinstance(evidence, str) and len(evidence) > 512:
                    body["evidence"] = view_with_limit(max(256, len(evidence) // 2))
                else:
                    disposable = [k for k in body if k not in ("evidence", "task", "catalog")
                                  and _json_size(body[k]) > _json_size({"truncated": True})]
                    if not disposable:
                        raise ValueError("structured task and complete tool catalog exceed model input budget")
                    largest = max(disposable, key=lambda k: _json_size(body[k]))
                    body[largest] = {"truncated": True}
            return body
        body = fit(body)
        raw = None
        try:
            raw = invoke(body)
            return role.normalize(raw)
        except (ValueError, TypeError, KeyError) as exc:
            repair = {**body, "repair_error": str(exc)[:500],
                      "instruction": "Return one concise valid JSON object matching the role schema. Limit each list to at most six items."}
            if raw is not None:
                repair["response_to_repair"] = _bounded(raw, 6000)
            raw = invoke(fit(repair), True)
            return role.normalize(raw)

    def view(ids=None):
        return ledger.render(ids, max_obs_chars=cfg.max_obs_chars, max_total_chars=cfg.max_total_evidence_chars)

    def view_with_limit(limit):
        return ledger.render(max_obs_chars=cfg.max_obs_chars, max_total_chars=limit)

    def update(role, value):
        if not cfg.enable_coevolution or role not in cfg.evolve_roles:
            return
        proposed = normalize_memory(role, value or {}, [entry["id"] for entry in ledger.summary()["observations"]], gated=cfg.evidence_gated_update)
        for key, items in proposed.items():
            memories[role][key] = items

    try:
        for current_round in range(1, cfg.max_rounds + 1):
            check()
            if current_round > 1:
                restore_state(initial_tool_snapshot)
            needs_new_probe = bool(tool_info and tool_calls < cfg.max_tool_calls and previous_vw
                                   and previous_vw.get("verdict") != "accept"
                                   and not _format_only(previous_vw))
            base = {"task": task_data, "round": current_round, "catalog": catalog,
                    "evidence": view(), "previous_candidate": previous,
                    "previous_work_review": previous_vw, "best_accepted": best,
                    "falsification_mode": best is not None,
                    "is_final_round": current_round == cfg.max_rounds,
                    "verification_requirements": {
                        "empirical_tool_required": bool(tool_info),
                        "new_probe_required": needs_new_probe,
                        "detail_inspection_tools": sorted(inspection_tools),
                        "public_context_evidence_id": "public:task"}}
            plan = ask(Planner, {**base, "memory": memories["planner"]})
            if current_round > 1:
                update("planner", plan.get("epistemic_update"))
            trace.append({"event": "plan", "round": current_round, "value": _bounded(plan, 6000)})
            vp = ask(PlanValidator, {**base, "plan": plan, "memory": memories["plan_validator"],
                                     "previous_plan_review": previous_vp})
            update("plan_validator", vp.get("epistemic_update"))
            trace.append({"event": "plan_review", "round": current_round, "value": _bounded(vp, 6000)})
            # The host never dispatches a plan step as a tool call. Invalid tool
            # names are surfaced to the worker, which can choose a registered tool.
            steps = vp.get("repaired_steps") or plan.get("steps", [])
            invalid = [s.get("tool_name") for s in steps if isinstance(s, dict) and s.get("tool_name") and s.get("tool_name") not in tool_info]
            if invalid:
                vp.setdefault("execution_warnings", []).append("Unregistered planned tools: " + ", ".join(map(str, invalid)))
            required_read_only_tools = list(dict.fromkeys(
                step["tool_name"] for step in steps if isinstance(step, dict)
                and step.get("required") is not False and step.get("tool_name") in tool_info
                and tool_policy[step["tool_name"]] == "read_only"))
            round_ids = []
            last_observation = None
            candidate = None
            synthesized_without_tools = False
            round_successes = []

            def empirical_gaps():
                successful = [entry for entry in ledger.summary()["observations"] if entry["ok"]]
                gaps = []
                has_terminal_coverage = any(a.get("terminal_failure") for rows in attempts_by_tool.values() for a in rows)
                if any(policy == "read_only" for policy in tool_policy.values()) and not successful and not has_terminal_coverage:
                    gaps.append("Execute a successful empirical tool call first; public task context alone does not verify a claim.")
                uncovered = [name for name in required_read_only_tools if not covered(name)]
                if uncovered:
                    gaps.append("Required read-only investigation is uncovered: " + ", ".join(uncovered))
                if round_ids and not ledger.get(round_ids[-1])["ok"] and not successful and not has_terminal_coverage:
                    gaps.append("The latest tool call failed. Repair and execute a corrected call before treating the investigation as complete.")
                if needs_new_probe and not successful and not round_successes and tool_calls < cfg.max_tool_calls:
                    gaps.append("The previous empirical audit failed. Execute a NEW discriminative probe with repaired or changed arguments this round; do not resubmit rejected evidence.")
                if inspection_tools and any(_has_previews(ledger.get(entry["id"]).get("result")) for entry in successful) and not any(entry["tool_name"] in inspection_tools for entry in successful):
                    gaps.append("Only retrieval previews are available. Inspect relevant full content using a detail tool before finalizing, then verify every required reasoning link.")
                return gaps

            def covered(name):
                entries = attempts_by_tool.get(name, [])
                if any(e["ok"] for e in entries):
                    return True
                terminal = [e for e in entries if e.get("terminal_failure")]
                return len(terminal) >= 2 and len({e["call_key"] for e in terminal}) >= 2

            for step in range(1, cfg.max_actions_per_round + 1):
                check()
                gaps = empirical_gaps()
                uncovered = [name for name in required_read_only_tools if not covered(name)]
                preview_blocked = bool(inspection_tools and
                    any(gap.startswith("Only retrieval previews") for gap in gaps))
                force_empirical = bool(any(p == "read_only" for p in tool_policy.values()) and not any(entry["ok"] for entry in ledger.summary()["observations"])
                                       and tool_calls < cfg.max_tool_calls)
                attempted_inspection = any(entry["tool_name"] in inspection_tools
                                           for entry in ledger.summary()["observations"])
                force_inspection = bool(preview_blocked and not attempted_inspection
                                        and tool_calls < cfg.max_tool_calls)
                cap_escape = tool_calls >= cfg.max_tool_calls or current_round >= 3
                gate_active = bool(uncovered and not cap_escape and tool_calls < cfg.max_tool_calls)
                packet = {**base, "evidence": view(), "plan": plan, "plan_review": vp,
                          "memory": memories["worker"], "last_observation": last_observation,
                          "actions_remaining": cfg.max_actions_per_round - step + 1,
                          "tool_calls_remaining": cfg.max_tool_calls - tool_calls,
                          "empirical_gaps": gaps, "required_tool_names": uncovered or None,
                          "terminal_failure_ids": sorted(a["id"] for rows in attempts_by_tool.values() for a in rows if a.get("terminal_failure")),
                          "coverage_ids": sorted({e["id"] for e in ledger.summary()["observations"] if e["ok"]} |
                                                 {a["id"] for rows in attempts_by_tool.values() for a in rows if a.get("terminal_failure")}),
                          "allow_final": not gate_active,
                          "planned_unexecuted_steps": [s for s in steps if isinstance(s, dict)
                              and s.get("required") is not False and s.get("tool_name") in tool_info
                              and not any(e["tool_name"] == s.get("tool_name") and e["ok"]
                                          for e in ledger.summary()["observations"])],
                          "force_empirical_tool": force_empirical and not uncovered,
                          "force_inspection_tool": force_inspection,
                          "inspection_tool_names": sorted(inspection_tools),
                          "preview_blocked": preview_blocked}
                action = ask(Worker, packet)
                if action["action"] == "final":
                    gaps = empirical_gaps()
                    successful_ids = {entry["id"] for entry in ledger.summary()["observations"] if entry["ok"]}
                    if tool_info and successful_ids and not (set(_ids(action)) & (successful_ids | {"public:task"})):
                        gaps.append("Cite the exact host-issued evidence IDs of successful observations supporting this answer; tool names and invented IDs are invalid.")
                    if gaps and tool_calls < cfg.max_tool_calls and not (uncovered and cap_escape):
                        last_observation = {"final_rejected": True, "instruction": " ".join(gaps),
                                            "remaining_actions": cfg.max_actions_per_round - step}
                        trace.append({"event": "premature_final_blocked", "round": current_round,
                                      "gaps": gaps})
                        continue
                    candidate = action
                    if uncovered and cap_escape:
                        candidate["unresolved_issues"] = list(candidate.get("unresolved_issues", [])) + [
                            "Required read-only checks remain uncovered: " + ", ".join(uncovered)]
                    break
                name, args = action["name"], action["arguments"]
                if name == "ledger_lookup":
                    eid = args.get("id")
                    last_observation = {"lookup": eid, "observation": ledger.window(eid, offset=args.get("offset", 0), max_chars=cfg.max_obs_chars)}
                    trace.append({"event": "ledger_lookup", "round": current_round, "id": eid})
                    continue
                call_key = (name, ledger.canonical_arguments(args))
                existing = ledger.find(name, args, failed_only=True)
                if existing is None and call_key not in mutated_call_keys and name in tool_info and (tool_info[name].get("mutates") is False or tool_info[name].get("read_only") is True):
                    existing = ledger.find(name, args)
                if existing is not None:
                    last_observation = {"reused_evidence_id": existing["id"], "observation": view([existing["id"]]),
                                        "instruction": "This exact call already ran and is reused, not a new empirical probe. If a new probe is required or the previous answer was rejected, change the arguments to test the disputed assumption or choose another relevant tool. A successful old call does not establish that the rejected answer was correct; repair failed arguments before retrying."}
                    trace.append({"event": "tool_reused", "round": current_round, "id": existing["id"]})
                    continue
                if tool_calls >= cfg.max_tool_calls:
                    last_observation = {"error": "Tool budget exhausted; synthesize from existing evidence."}
                    trace.append({"event": "tool_budget", "round": current_round})
                    break
                if name not in tool_info:
                    last_observation = {"error": "Unknown tool; select a registered tool."}
                    continue
                check()
                tool_calls += 1
                before_snapshot = snapshot_state()
                try:
                    output = tools.call(name, args)
                    entry = ledger.record(name, args, True, result=output, round=current_round, step=step)
                except Exception as exc:
                    details = {"message": str(exc)}
                    for field_name in ("status_code", "status", "code", "not_found"):
                        value = getattr(exc, field_name, None)
                        if value is not None:
                            details[field_name] = value
                    entry = ledger.record(name, args, False, error=details, round=current_round, step=step)
                after_snapshot = snapshot_state()
                if _snap_changed(before_snapshot, after_snapshot):
                    mutated_call_keys.add(call_key)
                round_ids.append(entry["id"])
                if name in required_read_only_tools:
                    terminal = _terminal_failure(entry)
                    attempts_by_tool.setdefault(name, []).append({
                        "id": entry["id"], "ok": entry["ok"], "call_key": call_key,
                        "terminal_failure": terminal, "arguments": copy.deepcopy(args)})
                    if terminal:
                        entry["terminal_failure"] = terminal
                if entry["ok"]:
                    round_successes.append(entry["id"])
                last_observation = {"evidence_id": entry["id"], "ok": entry["ok"],
                                    "observation": view([entry["id"]])}
                if not entry["ok"]:
                    last_observation["instruction"] = "Repair the failed call now using the public tool schema and context, then retry with corrected arguments within this round. An error is not evidence of absence; do not finalize from it."
                trace.append({"event": "tool_call", "round": current_round, "id": entry["id"], "ok": entry["ok"]})
            if candidate is None:
                synthesized_without_tools = True
                packet = {**base, "evidence": view(), "plan": plan, "plan_review": vp,
                          "memory": memories["worker"], "last_observation": last_observation,
                          "tools_disabled": True, "empirical_gaps": empirical_gaps(),
                          "instruction": "Return action final using available evidence; in the final round or when tools are disabled, synthesize the single best empirically supported answer and leave unresolved_issues empty when the evidence supports it. Preserve the exact public answer shape, including scalar array items and valid empty collections."}
                candidate = ask(Worker, packet)
                if candidate.get("action") != "final":
                    candidate = None
            if candidate is None:
                raise _Budget("worker did not synthesize")
            update("worker", candidate.get("epistemic_update") or candidate.get("tactic_notes"))
            public_ids = {"public:task"}
            candidate["cited_evidence_ids"] = ledger.valid_ids(_ids(candidate)) + sorted(set(_ids(candidate)) & public_ids)
            host_gaps = empirical_gaps()
            successful_ids = {entry["id"] for entry in ledger.summary()["observations"] if entry["ok"]}
            terminal_failure_ids = {attempt["id"] for entries in attempts_by_tool.values()
                                    for attempt in entries if attempt.get("terminal_failure")}
            coverage_ids = successful_ids | terminal_failure_ids
            if synthesized_without_tools and successful_ids and not (set(candidate["cited_evidence_ids"]) & successful_ids):
                candidate["cited_evidence_ids"] = [sorted(successful_ids)[-1], *candidate["cited_evidence_ids"]]
            candidate_snapshot = snapshot_state()
            trace.append({"event": "candidate", "round": current_round, "value": _bounded(candidate, 6000)})
            vw = ask(WorkerValidator, {**base, "evidence": view(), "plan": plan, "plan_review": vp,
                                       "candidate": candidate, "empirical_gaps": host_gaps,
                                       "round_evidence_ids": round_ids,
                                       "memory": memories["worker_validator"]})
            update("worker_validator", vw.get("epistemic_update"))
            trace.append({"event": "work_review", "round": current_round, "value": _bounded(vw, 6000)})
            review_ids = set(ledger.valid_ids(_ids({"cited_evidence_ids": vw.get("cited_evidence_ids", []),
                                                    "verified_claims": vw.get("verified_claims", []),
                                                    "epistemic_update": {"verified_claims": vw.get("epistemic_update", {}).get("verified_claims", [])}})))
            review_ids.update(set(_ids(vw)) & public_ids)
            candidate_ids = set(candidate["cited_evidence_ids"])
            successful_ids = {entry["id"] for entry in ledger.summary()["observations"] if entry["ok"]}
            candidate_ids = set(candidate["cited_evidence_ids"])
            grounded = (not tool_info) or bool(successful_ids and candidate_ids & successful_ids)
            candidate_covered = (not tool_info) or bool(coverage_ids and candidate_ids & coverage_ids)
            preview_blocked = any(gap.startswith("Only retrieval previews") for gap in host_gaps)
            accepted = (vw.get("verdict") == "accept" and candidate.get("answer") is not None
                        and grounded and not host_gaps and not candidate.get("unresolved_issues")
                        and not vw.get("unresolved_issues"))
            if not accepted:
                # A model's accept verdict cannot hide a host rejection from
                # the next round, including a citation-only repair at the cap.
                gate_reasons = list(host_gaps)
                if not grounded:
                    gate_reasons.append("Candidate and audit must cite shared successful host-issued evidence IDs supporting the answer.")
                if candidate.get("answer") is None:
                    gate_reasons.append("The candidate answer is null; provide a supported answer or retain explicit uncertainty.")
                if candidate.get("unresolved_issues"):
                    gate_reasons.append("The candidate still reports unresolved issues: " + str(candidate["unresolved_issues"]))
                if vw.get("verdict") == "accept" and vw.get("unresolved_issues"):
                    gate_reasons.append("The audit cannot accept while reporting unresolved issues.")
                if gate_reasons:
                    citation_only = (not grounded and not host_gaps and candidate.get("answer") is not None
                                     and not candidate.get("unresolved_issues") and not vw.get("unresolved_issues")
                                     and vw.get("verdict") == "accept")
                    vw["verdict"] = "revise"
                    vw["evidence_complete"] = citation_only
                    if citation_only:
                        vw["format_fix_instruction"] = "Repair candidate/audit citations using the exact successful evidence IDs already in the ledger; preserve the supported answer value."
                    else:
                        vw["unresolved_issues"] = list(vw.get("unresolved_issues", [])) + gate_reasons
                    trace.append({"event": "empirical_gate", "round": current_round,
                                  "gaps": gate_reasons, "citation_only": citation_only})
            if accepted:
                changed = best is not None and candidate["answer"] != best["answer"]
                support_ids = candidate_ids & successful_ids
                if not changed or not cfg.evidence_gated_update or bool(support_ids or ((review_ids & candidate_ids) - best_ids)):
                    best = copy.deepcopy(candidate)
                    best_snapshot = copy.deepcopy(candidate_snapshot)
                    best_ids = candidate_ids & successful_ids
                else:
                    trace.append({"event": "regression_blocked", "round": current_round})
            if candidate.get("answer") is not None and candidate_covered and not preview_blocked:
                if best is None or accepted or best_grounded is None or bool(candidate_ids & successful_ids):
                    best_grounded = copy.deepcopy(candidate)
                    best_grounded_snapshot = copy.deepcopy(candidate_snapshot)
            rounds = current_round
            history.append({"round": current_round, **copy.deepcopy(memories)})
            previous, previous_vp, previous_vw = candidate, vp, vw
            if accepted and best is not None and cfg.allow_early_stop and current_round >= cfg.min_rounds:
                early = current_round < cfg.max_rounds
                break
        status = "completed" if best is not None or best_grounded is not None else "budget_exhausted"
    except (_Budget, ModelBudgetError) as exc:
        trace.append({"event": "budget_exhausted", "reason": str(exc)})
        if isinstance(exc, ModelBudgetError):
            error = f"{type(exc).__name__}: {exc}"
        coverage_entries = [e for e in ledger.summary()["observations"]
                            if e["ok"] or e["id"] in {a["id"] for rows in attempts_by_tool.values() for a in rows if a.get("terminal_failure")}]
        if best is None and best_grounded is None and coverage_entries:
            try:
                check(reserved=True)
                if calls < cfg.max_model_calls:
                    fallback_payload = {"task": task_data, "round": max(1, rounds), "catalog": catalog,
                        "evidence": view(), "previous_candidate": previous, "previous_work_review": previous_vw,
                        "tools_disabled": True, "is_final_round": True,
                        "instruction": "Synthesize the best supported final answer from successful observations."}
                    fallback = ask(Worker, fallback_payload, reserved=True)
                    if fallback.get("action") == "final":
                        fallback["cited_evidence_ids"] = ledger.valid_ids(_ids(fallback))
                        coverage = sorted(e["id"] for e in coverage_entries)
                        if coverage and not set(fallback["cited_evidence_ids"]) & set(coverage):
                            fallback["cited_evidence_ids"] = [coverage[-1], *fallback["cited_evidence_ids"]]
                        best_grounded = fallback
                        best_grounded_snapshot = snapshot_state()
            except Exception:
                pass
        status = "completed" if best is not None or best_grounded is not None else "budget_exhausted"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        trace.append({"event": "error", "reason": error})
        status = "completed" if best is not None or best_grounded is not None else "error"
    chosen = best if best is not None else best_grounded
    chosen_snapshot = best_snapshot if best is not None else best_grounded_snapshot
    try:
        restore_state(chosen_snapshot if chosen is not None else initial_tool_snapshot)
    except Exception as exc:
        error = error or f"{type(exc).__name__}: {exc}"
        if chosen is None:
            status = "error"
    completion_mode = "strict_accept" if best is not None else ("grounded_round_commit" if best_grounded is not None else None)
    final_coverage_ids = {e["id"] for e in ledger.summary()["observations"] if e["ok"]}
    final_coverage_ids.update(a["id"] for rows in attempts_by_tool.values() for a in rows if a.get("terminal_failure"))
    artifacts = {"rounds_completed": rounds, "early_stopped": early, "model_calls": calls,
                 "tool_calls": tool_calls, "elapsed_seconds": time.monotonic() - started,
                 "role_memories_history": history, "ledger_summary": ledger.summary(),
                 "best_accepted": best, "best_grounded": best_grounded,
                 "completion_mode": completion_mode, "tool_policy": tool_policy,
                 "coverage_ids": sorted(final_coverage_ids),
                 "error": error}
    return Result(method=cfg.method_name, status=status, answer=copy.deepcopy(chosen["answer"]) if chosen else None,
                  trace=trace, artifacts=artifacts)
