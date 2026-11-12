"""Role schemas and task-local epistemic memory normalization.

The host invokes models and owns the evidence ledger; these helpers never make
model or tool calls and cannot establish that a model's claim is true.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from . import prompts


@dataclass
class PlannerMemory:
    verified_anchors: list = field(default_factory=list)
    falsified_paths: list = field(default_factory=list)
    open_gaps: list = field(default_factory=list)


@dataclass
class PlanValidatorMemory:
    learned_guardrails: list = field(default_factory=list)
    acceptance_predicates: list = field(default_factory=list)


@dataclass
class WorkerMemory:
    effective_tactics: list = field(default_factory=list)
    avoid_patterns: list = field(default_factory=list)


@dataclass
class WorkerValidatorMemory:
    verified_claims: list = field(default_factory=list)
    rejected_noise: list = field(default_factory=list)
    diagnostic_probes: list = field(default_factory=list)


MEMORY_TYPES = {"planner": PlannerMemory, "plan_validator": PlanValidatorMemory,
                "worker": WorkerMemory, "worker_validator": WorkerValidatorMemory}
LIMITS = {"verified_anchors": 6, "falsified_paths": 6, "open_gaps": 6,
          "learned_guardrails": 5, "acceptance_predicates": 6,
          "effective_tactics": 5, "avoid_patterns": 5,
          "verified_claims": 6, "rejected_noise": 6, "diagnostic_probes": 6}


def empty_memory(role: str) -> dict[str, list]:
    return asdict(MEMORY_TYPES[role]())


def string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def as_list(value: Any) -> list:
    if value is None:
        return []
    return deepcopy(value) if isinstance(value, list) else [deepcopy(value)]


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "accept", "approved", "yes"):
            return True
        if normalized in ("false", "revise", "rejected", "no"):
            return False
    raise ValueError("expected boolean")


def object_value(value: Any) -> dict:
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("```"):
            value = value.partition("\n")[2].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("expected JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return deepcopy(value)


def normalize_memory(role: str, update: Any, valid_ids: Any, gated: bool = True) -> dict:
    """Return bounded replacement lists; absent fields are left absent.

    When gated, learned entries require at least one host-owned evidence ID.
    Acceptance predicates are prospective criteria, so they need no evidence.
    The caller chooses whether to merge or replace prior memory.
    """
    if not isinstance(update, dict):
        return {}
    valid = set(valid_ids)
    result = {}
    for key in empty_memory(role):
        if key not in update:
            continue
        entries = []
        for item in as_list(update[key]):
            if isinstance(item, dict):
                item = deepcopy(item)
                ids = [str(e) for e in as_list(item.get("evidence_ids", [])) if str(e) in valid]
                item["evidence_ids"] = list(dict.fromkeys(ids))
            else:
                item = {"text": string(item), "evidence_ids": []}
            if gated and key != "acceptance_predicates" and not item["evidence_ids"]:
                continue
            entries.append(item)
        result[key] = entries[-LIMITS[key]:]
    return result


def _update(value: dict) -> dict:
    update = value.get("epistemic_update", {})
    return update if isinstance(update, dict) else {}


class Planner:
    name = "planner"
    prompt = prompts.PLANNER
    response_schema = {"type": "object", "properties": {
        "epistemic_update": {"type": "object"},
        "goal": {"type": "string"}, "output_contract": {"type": "string"},
        "steps": {"type": "array", "minItems": 1, "maxItems": 5, "items": {
            "type": "object", "properties": {"step_id": {"type": "string"},
                "sub_goal": {"type": "string"}, "purpose": {"type": "string"},
                "tool_name": {"type": ["string", "null"]}, "required": {"type": "boolean"}},
            "required": ["sub_goal"], "additionalProperties": False}}},
        "required": ["goal", "output_contract", "steps"], "additionalProperties": False}

    @classmethod
    def normalize(cls, value: Any) -> dict:
        value = object_value(value)
        if isinstance(value.get("plan"), dict):
            value = {**value, **value["plan"]}
        if "goal" not in value or "output_contract" not in value or not as_list(value.get("steps")):
            raise ValueError("plan requires goal, output_contract and steps")
        steps = []
        for i, step in enumerate(as_list(value["steps"])[:5], 1):
            if not isinstance(step, dict) or not (step.get("sub_goal") or step.get("goal")):
                raise ValueError("plan step requires sub_goal")
            tool = step.get("tool_name", step.get("tool"))
            steps.append({"step_id": string(step.get("step_id", i)),
                          "sub_goal": string(step.get("sub_goal", step.get("goal"))),
                          "tool_name": None if tool is None else string(tool),
                          "purpose": string(step.get("purpose", "")),
                          "required": step.get("required") is not False})
        return {"epistemic_update": _update(value), "goal": string(value["goal"]),
                "output_contract": string(value["output_contract"]), "steps": steps}


class PlanValidator:
    name = "plan_validator"
    prompt = prompts.PLAN_VALIDATOR
    _text = {"type": "string", "maxLength": 600}
    _text_list = {"type": "array", "maxItems": 6, "items": _text}
    _memory_entry = {"anyOf": [_text, {"type": "object", "properties": {
        **{key: {"type": "string", "maxLength": 600}
           for key in ("text", "claim", "rule", "predicate", "reason")},
        "evidence_ids": {"type": "array", "maxItems": 16,
                         "items": {"type": "string", "maxLength": 64}}},
        "additionalProperties": False}]}
    _memory_list = {"type": "array", "maxItems": 6, "items": _memory_entry}
    _repaired_steps = deepcopy(Planner.response_schema["properties"]["steps"])
    # An empty repair means execute the original plan, rather than an invalid
    # empty Planner plan. Retain the Planner's closed step keys and upper bound.
    _repaired_steps["minItems"] = 0
    for _field, _schema in _repaired_steps["items"]["properties"].items():
        if _schema.get("type") == "string" or isinstance(_schema.get("type"), list) and "string" in _schema["type"]:
            _schema["maxLength"] = 64 if _field == "step_id" else 128 if _field == "tool_name" else 600
    response_schema = {"type": "object", "properties": {
        "approved": {"type": "boolean"},
        "epistemic_update": {"type": "object", "properties": {
            "learned_guardrails": _memory_list, "acceptance_predicates": _memory_list},
            "additionalProperties": False},
        "acceptance_predicates": _text_list,
        "execution_guardrails": _text_list,
        "repaired_steps": _repaired_steps,
        "execution_brief": {"type": "string", "maxLength": 1200},
        "execution_warnings": _text_list,
        "must_check": _text_list,
        "probe_instruction": _text},
        "required": ["approved"], "additionalProperties": False}

    @classmethod
    def normalize(cls, value: Any) -> dict:
        value = object_value(value)
        if "approved" not in value:
            raise ValueError("plan review requires approved")
        return {"epistemic_update": _update(value), "approved": boolean(value["approved"]),
                "acceptance_predicates": as_list(value.get("acceptance_predicates")),
                "execution_guardrails": as_list(value.get("execution_guardrails")),
                "repaired_steps": as_list(value.get("repaired_steps"))[:5],
                "execution_brief": string(value.get("execution_brief", "")),
                "execution_warnings": as_list(value.get("execution_warnings")),
                "must_check": as_list(value.get("must_check")),
                "probe_instruction": string(value.get("probe_instruction", ""))}


class Worker:
    name = "worker"
    prompt = prompts.WORKER
    response_schema = {"type": "object", "anyOf": [
        {"type": "object", "properties": {"action": {"type": "string", "enum": ["tool"]},
            "name": {"type": "string", "minLength": 1}, "arguments": {"type": "object"},
            "reason": {"type": "string"}},
         "required": ["action", "name", "arguments"], "additionalProperties": False},
        {"type": "object", "properties": {"action": {"type": "string", "enum": ["final"]},
            "answer": {}, "item_evidence_map": {"type": "object"},
            "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
            "unresolved_issues": {"type": "array", "items": {}},
            "tactic_notes": {"type": "object"}, "epistemic_update": {"type": "object"}},
         "required": ["action", "answer"], "additionalProperties": False}]}

    @classmethod
    def constrained_schema(cls, catalog: list[dict], *, tools_disabled: bool = False,
                           required_tool_names: list[str] | None = None,
                           allow_final: bool = False) -> dict:
        """Constrain action type without selecting a tool or its arguments."""
        if tools_disabled:
            return deepcopy(cls.response_schema["anyOf"][1])
        branches = []
        allowed = set(required_tool_names) if required_tool_names is not None else None
        for tool in catalog:
            if allowed is not None and tool["name"] not in allowed:
                continue
            branches.append({"type": "object", "properties": {
                "action": {"type": "string", "enum": ["tool"]},
                "name": {"type": "string", "enum": [tool["name"]]},
                "arguments": deepcopy(tool.get("parameters", {"type": "object"})),
                "reason": {"type": "string"}},
                "required": ["action", "name", "arguments"], "additionalProperties": False})
        if allow_final:
            branches.append(deepcopy(cls.response_schema["anyOf"][1]))
        if not branches:
            raise ValueError("Required empirical action has no registered tool")
        return branches[0] if len(branches) == 1 else {"type": "object", "anyOf": branches}

    @classmethod
    def normalize(cls, value: Any) -> dict:
        value = object_value(value)
        action = value.get("action")
        if action == "tool":
            if not isinstance(value.get("name"), str) or not isinstance(value.get("arguments"), dict):
                raise ValueError("tool action requires name and arguments object")
            return {"action": "tool", "name": value["name"], "arguments": value["arguments"],
                    "reason": string(value.get("reason", ""))}
        if action == "final":
            if "answer" not in value:
                raise ValueError("final action requires answer")
            return {"action": "final", "answer": value["answer"],
                    "item_evidence_map": value.get("item_evidence_map", {}),
                    "cited_evidence_ids": as_list(value.get("cited_evidence_ids")),
                    "unresolved_issues": as_list(value.get("unresolved_issues")),
                    "tactic_notes": value.get("tactic_notes", {}),
                    "epistemic_update": _update(value)}
        raise ValueError("worker action must be tool or final")


class WorkerValidator:
    name = "worker_validator"
    prompt = prompts.WORKER_VALIDATOR
    response_schema = {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["accept", "revise"]},
        "epistemic_update": {"type": "object"},
        "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "verified_claims": {"type": "array", "items": {}},
        "rejected_noise": {"type": "array", "items": {}},
        "unresolved_issues": {"type": "array", "items": {}},
        "evidence_complete": {"type": "boolean"},
        "format_fix_instruction": {"type": "string"},
        "diagnostic_task_for_planner": {"type": "object"},
        "reason": {"type": "string", "maxLength": 1200}},
        "required": ["verdict"], "additionalProperties": False}

    @classmethod
    def normalize(cls, value: Any) -> dict:
        value = object_value(value)
        if "verdict" not in value:
            raise ValueError("work review requires verdict")
        verdict = "accept" if boolean(value["verdict"]) else "revise"
        return {"epistemic_update": _update(value), "verdict": verdict,
                "cited_evidence_ids": as_list(value.get("cited_evidence_ids")),
                "verified_claims": as_list(value.get("verified_claims")),
                "rejected_noise": as_list(value.get("rejected_noise")),
                "unresolved_issues": as_list(value.get("unresolved_issues")),
                "evidence_complete": value.get("evidence_complete") is True,
                "format_fix_instruction": string(value.get("format_fix_instruction", "")),
                "diagnostic_task_for_planner": value.get("diagnostic_task_for_planner", {}),
                "reason": string(value.get("reason", ""))}
