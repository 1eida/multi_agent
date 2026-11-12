"""Minimal task, model, tool, and optional OpenAI-compatible interfaces."""
from __future__ import annotations
import copy, json, math, os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Protocol

@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    context: Any = None

@dataclass
class Result:
    method: str
    status: str
    answer: Any = None
    trace: list[dict] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    def as_dict(self): return copy.deepcopy(asdict(self))

class Model(Protocol):
    def ask(self, system: str, payload: dict, *, role: str, json_mode: bool = True) -> Any: ...

class ModelBudgetError(RuntimeError):
    """A declared model resource limit was exhausted."""

@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    function: Callable[..., Any]
    mutates: bool = False

class Toolbox:
    def __init__(self, tools=(), *, snapshot=None, restore=None):
        values = list(tools.values()) if isinstance(tools, dict) else list(tools)
        self.tools = {tool.name: tool for tool in values}
        if len(self.tools) != len(values): raise ValueError("tool names must be unique")
        if (snapshot is None) != (restore is None): raise ValueError("snapshot and restore callbacks must be paired")
        self._snapshot, self._restore = snapshot, restore
    def catalog(self):
        return [{"name": t.name, "description": t.description, "parameters": copy.deepcopy(t.parameters), "mutates": t.mutates} for t in self.tools.values()]
    def call(self, name, arguments):
        if name not in self.tools: raise ValueError(f"unknown tool: {name}")
        if not isinstance(arguments, dict): raise ValueError("tool arguments must be an object")
        return self.tools[name].function(**copy.deepcopy(arguments))
    def snapshot(self):
        if self._snapshot is not None: return self._snapshot()
        if any(t.mutates for t in self.tools.values()): raise RuntimeError("mutating tools require snapshot/restore callbacks")
        return None
    def restore(self, snapshot):
        if self._restore is not None: self._restore(snapshot)
        elif any(t.mutates for t in self.tools.values()): raise RuntimeError("cannot roll back mutable tool state")

@dataclass(frozen=True)
class ModelConfig:
    base_url: str | None = None
    model: str = "model-name"
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: float = 120.0
    def __post_init__(self):
        if self.max_tokens < 1 or self.timeout <= 0: raise ValueError("max_tokens and timeout must be positive")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2: raise ValueError("temperature must be between 0 and 2")

class OpenAIModel:
    """Optional OpenAI-compatible transport; install the openai extra to use it."""
    def __init__(self, config=None, *, api_key=None):
        try: from openai import OpenAI
        except ImportError as exc: raise ImportError("Install four-agent-method[openai] to use OpenAIModel") from exc
        base = config or ModelConfig()
        self.config = replace(base, base_url=base.base_url or os.environ.get("OPENAI_BASE_URL"),
                              model=os.environ.get("OPENAI_MODEL", base.model))
        self.client = OpenAI(base_url=self.config.base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"), max_retries=0, timeout=self.config.timeout)
    def _ask(self, system, payload, role, schema=None, json_mode=True):
        args = {"model": self.config.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], "max_tokens": self.config.max_tokens, "temperature": self.config.temperature}
        if schema is None and json_mode: args["response_format"] = {"type": "json_object"}
        else: args["response_format"] = {"type": "json_schema", "json_schema": {"name": "role_response", "strict": True, "schema": schema}}
        result = self.client.chat.completions.create(**args)
        if not result.choices or not result.choices[0].message.content: raise ValueError(f"model returned no content for {role}")
        return json.loads(result.choices[0].message.content) if json_mode or schema is not None else result.choices[0].message.content
    def ask(self, system, payload, *, role, json_mode=True):
        return self._ask(system, payload, role, json_mode=json_mode)
    def ask_structured(self, system, payload, *, role, response_schema):
        return self._ask(system, payload, role, response_schema)
    def close(self): self.client.close()
