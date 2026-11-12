"""Host-owned tool observations and bounded, provenance-aware views."""

from __future__ import annotations

import copy
import json
from typing import Any, Iterable


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


class EvidenceLedger:
    """Append-only records of actual tool executions, scoped to a single task.

    Callers decide whether a tool is read-only before reusing a successful
    result. Failed executions may be looked up separately via ``failed_only``.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_call: dict[tuple[str, str], list[str]] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def canonical_arguments(arguments: Any) -> str:
        return _json(arguments)

    def record(self, tool_name: str, arguments: Any, ok: bool, result: Any = None,
               error: Any = None, *, round: int, step: int) -> dict[str, Any]:
        eid = f"e{len(self._entries) + 1}"
        entry = {"id": eid, "tool_name": str(tool_name),
                 "arguments": copy.deepcopy(arguments), "ok": bool(ok),
                 "round": round, "step": step}
        entry["result" if ok else "error"] = copy.deepcopy(result if ok else error)
        self._entries.append(entry)
        self._by_id[eid] = entry
        key = (str(tool_name), self.canonical_arguments(arguments))
        self._by_call.setdefault(key, []).append(eid)
        return copy.deepcopy(entry)

    def get(self, eid: str) -> dict[str, Any] | None:
        entry = self._by_id.get(eid)
        return copy.deepcopy(entry) if entry is not None else None

    def find(self, tool_name: str, arguments: Any, *, failed_only: bool = False) -> dict[str, Any] | None:
        key = (str(tool_name), self.canonical_arguments(arguments))
        for eid in reversed(self._by_call.get(key, [])):
            entry = self._by_id[eid]
            if not failed_only or not entry["ok"]:
                return copy.deepcopy(entry)
        return None

    def valid_ids(self, ids: Any) -> list[str]:
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, (list, tuple, set)):
            return []
        seen: set[str] = set()
        result: list[str] = []
        for eid in ids:
            if isinstance(eid, str) and eid in self._by_id and eid not in seen:
                seen.add(eid)
                result.append(eid)
        return result

    def window(self, eid: str, *, offset: int = 0, max_chars: int = 4000) -> dict[str, Any] | None:
        """Read a bounded window of a stored result's canonical JSON text."""
        entry = self._by_id.get(eid)
        if entry is None:
            return None
        value = _json(entry.get("result") if entry["ok"] else entry.get("error"))
        start = min(max(0, int(offset)), len(value))
        end = min(start + max(0, int(max_chars)), len(value))
        return {"id": eid, "offset": start, "next_offset": end,
                "original_chars": len(value), "preview": value[start:end],
                "truncated": start > 0 or end < len(value)}

    def render(self, ids: Iterable[str] | None = None, *, max_obs_chars: int = 4000,
               max_total_chars: int = 24000) -> str:
        """Return a deterministic JSON view with hard character limits.

        Observation previews are JSON-encoded strings, preserving the raw
        value's representation until explicitly truncated. The original size,
        truncation flag, and omitted count make context loss visible.
        """
        limit = max(0, int(max_total_chars))
        if limit < 2:
            return ""[:limit]
        chosen = self._entries if ids is None else [self._by_id[eid] for eid in self.valid_ids(ids if isinstance(ids, str) else list(ids))]
        payload: dict[str, Any] = {"observations": [], "omitted": len(chosen)}
        if len(_json(payload)) > limit:
            return "{}"
        for entry in chosen:
            value = _json(entry.get("result") if entry["ok"] else entry.get("error"))
            args = _json(entry["arguments"])
            preview_cap = max(0, int(max_obs_chars))
            item = {"id": entry["id"], "tool_name": entry["tool_name"][:128],
                    "arguments_preview": args[:512], "arguments_chars": len(args),
                    "ok": entry["ok"], "round": entry["round"], "step": entry["step"],
                    "preview": "", "original_chars": len(value), "truncated": bool(value)}
            payload["observations"].append(item)
            payload["omitted"] -= 1
            if len(_json(payload)) > limit:
                payload["observations"].pop()
                payload["omitted"] += 1
                break
            lo, hi = 0, min(preview_cap, len(value))
            while lo < hi:
                mid = (lo + hi + 1) // 2
                item["preview"] = value[:mid]
                item["truncated"] = mid < len(value)
                if len(_json(payload)) <= limit:
                    lo = mid
                else:
                    hi = mid - 1
            item["preview"] = value[:lo]
            item["truncated"] = lo < len(value)
        return _json(payload)

    def summary(self) -> dict[str, Any]:
        return {"count": len(self._entries), "observations": [
            {"id": e["id"], "tool_name": e["tool_name"], "ok": e["ok"],
             "round": e["round"], "step": e["step"]} for e in self._entries]}
