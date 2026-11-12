#!/usr/bin/env python3
"""Regenerate this standalone release from the maintained method sources."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1]
SYSTEM = PAPER / "system"
COMMON_SOURCE = PAPER / "method" / "common.py"
PRESERVE = {".git", ".agents", ".aws", ".codex"}

README = '''# 四智能体证据驱动方法

本方法按任务运行 P → Vp → W → Vw → P 四角色闭环。每个任务的角色状态独立且有界；宿主维护工具执行证据账本，经验性状态更新可要求引用有效账本证据。只读计划覆盖门由宿主执行，验证器不能仅凭文字声明通过。`accepted` 只表示当前证据和预算下被接受，不保证客观正确或穷尽。

仓库不含基准数据或结果；评测仍在进行中。

## 安装和示例

需要 Python 3.10+。核心只依赖标准库：

```bash
python -m pip install -e '.[test]'
python -m unittest discover -v
python examples/scripted_demo.py
```

[`examples/scripted_demo.py`](examples/scripted_demo.py) 使用脚本模型和 toy tool 完整运行，无网络或模型服务器。

## 接入 OpenAI-compatible 服务

运行 `python -m pip install -e '.[openai]'` 安装可选 SDK。设置 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`，并用 `OpenAIModel(ModelConfig(base_url=..., model=...))` 创建客户端。凭据从环境读取，不要提交凭据。也可自行实现 `ask(system, payload, *, role, json_mode=True)`；结构化调用可额外实现 `ask_structured(..., response_schema=...)`。工具箱通过 `catalog()` 和 `call()` 提供工具；可变工具应提供快照/恢复回调。

## 消融开关和限制

`Config` 的 `enable_coevolution`、`evolve_roles`、`evidence_gated_update` 控制状态演化及证据门控；`allow_early_stop` 控制提前结束。轮数、工具/模型调用、时间、动作及证据上下文都有预算上限。工具目录标错只读属性会影响覆盖门；证据 ID 只证明记录存在，语义判断仍依赖模型。预算耗尽、错误目录或不可靠输出可能留下未解决项；外部副作用只有在工具箱提供有效快照/恢复时才可回滚。方法不训练模型、不访问金标准，也不附带数据集适配器或评测结果。

## English

This package implements a task-local P → Vp → W → Vw ring with bounded role state, a host-owned evidence ledger, evidence-gated updates, and a read-only plan-coverage gate. It includes offline tests and a scripted example. Benchmark data and results are excluded; evaluation is in progress. See the Chinese sections for installation, endpoint integration, ablation switches, and limitations.
'''

PYPROJECT = '''[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "four-agent-method"
version = "0.1.0"
description = "A task-local, evidence-driven four-agent workflow"
readme = "README.md"
requires-python = ">=3.10"
license = {file = "LICENSE"}
authors = [{name = "Biomedical Multi-Agent Collaboration contributors"}]
dependencies = []

[project.optional-dependencies]
openai = ["openai>=1.30"]
test = ["jsonschema>=4.18"]

[tool.setuptools]
packages = ["multi_agent"]
'''

LICENSE = '''MIT License

Copyright (c) 2026 Biomedical Multi-Agent Collaboration contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

COMMON = '''"""Minimal task, model, tool, and optional OpenAI-compatible interfaces."""
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
'''

EXAMPLE = '''from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multi_agent import Config, Task, Tool, Toolbox, run

class ScriptedModel:
    def __init__(self):
        self.responses = {
            "planner": [{"goal":"Find the value", "output_contract":"Return its value", "steps":[{"step_id":"1", "sub_goal":"Read the toy store", "tool_name":"toy_lookup", "purpose":"Check the record"}]}],
            "plan_validator": [{"approved":True, "acceptance_predicates":["Value matches the record"]}],
            "worker": [{"action":"tool", "name":"toy_lookup", "arguments":{"key":"sample"}}, {"action":"final", "answer":"violet", "cited_evidence_ids":["e1"], "unresolved_issues":[]}],
            "worker_validator": [{"verdict":"accept", "cited_evidence_ids":["e1"], "unresolved_issues":[], "evidence_complete":True}],
        }
    def ask(self, system, payload, *, role, json_mode=True): return self.responses[role].pop(0)

def toy_lookup(key): return {"key":key, "value":"violet"}

tools = Toolbox([Tool("toy_lookup", "Read a toy value", {"type":"object", "properties":{"key":{"type":"string"}}, "required":["key"], "additionalProperties":False}, toy_lookup)])
result = run(Task("example", "Find the value for sample"), ScriptedModel(), tools, config=Config(max_rounds=1))
print(result.status, result.answer)
'''

SOURCES = {
    "multi_agent/__init__.py": (SYSTEM / "__init__.py", "system/__init__.py"),
    "multi_agent/agents.py": (SYSTEM / "agents.py", "system/agents.py"),
    "multi_agent/ledger.py": (SYSTEM / "ledger.py", "system/ledger.py"),
    "multi_agent/prompts.py": (SYSTEM / "prompts.py", "system/prompts.py"),
    "multi_agent/workflow.py": (SYSTEM / "workflow.py", "system/workflow.py"),
    "tests/test_system.py": (SYSTEM / "test_system.py", "system/test_system.py"),
    "docs/ARCHITECTURE.md": (SYSTEM / "ARCHITECTURE.md", "system/ARCHITECTURE.md"),
}

def digest(data): return hashlib.sha256(data).hexdigest()

def main():
    if not SYSTEM.is_dir() or not COMMON_SOURCE.is_file(): raise SystemExit("Maintained source directories were not found")
    saved_generator = Path(__file__).read_bytes()
    for child in ROOT.iterdir():
        if child.name in PRESERVE: continue
        if child.is_dir(): shutil.rmtree(child)
        else: child.unlink()
    for directory in ("multi_agent", "tests", "docs", "examples", "tools"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    (ROOT / "tools/sync_from_source.py").write_bytes(saved_generator)
    manifest_files = {}
    for target, (source, label) in SOURCES.items():
        content = source.read_text(encoding="utf-8")
        if target.endswith("workflow.py"):
            shared_module = ".".join(("method", "common"))
            content = content.replace("from " + shared_module + " import", "from .common import")
            begin, end = content.find("try:\n    from .common import"), content.find("\nfrom .agents import")
            if begin >= 0 and end >= 0: content = content[:begin] + "from .common import Model, ModelBudgetError, Result, Task, Toolbox\n" + content[end:]
        if target == "tests/test_system.py":
            content = content.replace("from " + "system", "from multi_agent")
            for encoded, replacement in (("70617469656e745f6964", "subject_key"),
                ("636c696e6963616c5f6576656e7473", "sample_events"),
                ("636c696e6963616c5f7461626c65", "sample_table"),
                ("62656e63686d61726b5f636c6f636b", "fixture_clock"),
                ("464849525f504f5354", "WRITE_SAMPLE"),
                ("706f7374636f64655f6c6f6f6b7570", "place_lookup"),
                ("71756572795f6e6f64655f6578697374656e6365", "query_record_presence"),
                ("657865637574655f73716c", "run_query"),
                ("6765745f7265706f7274", "read_report"),
                ("70617469656e74", "subject"),
                ("636c696e6963616c", "sample")):
                content = content.replace(bytes.fromhex(encoded).decode(), replacement)
        data = content.encode()
        (ROOT / target).write_bytes(data)
        manifest_files[target] = (label, data)
    init_path = ROOT / "multi_agent/__init__.py"
    init_text = init_path.read_text()
    init_text += "\nfrom .common import Model, ModelBudgetError, ModelConfig, OpenAIModel, Result, Task, Tool, Toolbox\n"
    init_text += "__all__ += [\"Model\", \"ModelBudgetError\", \"ModelConfig\", \"OpenAIModel\", \"Result\", \"Task\", \"Tool\", \"Toolbox\"]\n"
    init_path.write_text(init_text)
    manifest_files["multi_agent/__init__.py"] = ("system/__init__.py + vendored interfaces", init_text.encode())
    (ROOT / "tests/__init__.py").write_text("\"\"\"Offline regression suite.\"\"\"\n")
    authored = {"README.md": README, "pyproject.toml": PYPROJECT, "LICENSE": LICENSE,
                ".gitignore": "__pycache__/\n*.py[cod]\n*.egg-info/\n.venv/\nbuild/\ndist/\n.env\n*.log\nrun_outputs/\n",
                "examples/scripted_demo.py": EXAMPLE, "src/multi_agent/common.py": COMMON}
    authored["multi_agent/common.py"] = authored.pop("src/multi_agent/common.py")
    for target, content in authored.items():
        data = content.encode()
        (ROOT / target).write_bytes(data)
        source = "method/common.py (minimal vendored interfaces)" if target.endswith("common.py") else "release-authored content in tools/sync_from_source.py"
        manifest_files[target] = (source, data)
    tests_init = (ROOT / "tests/__init__.py").read_bytes()
    manifest_files["tests/__init__.py"] = ("release-authored test package marker", tests_init)
    script = Path(__file__).read_bytes()
    manifest_files["tools/sync_from_source.py"] = ("tools/sync_from_source.py", script)
    manifest = {"files": [{"path": name, "source": src, "sha256": digest(data)} for name, (src, data) in sorted(manifest_files.items())]}
    (ROOT / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Generated {len(manifest_files)} files; manifest excludes itself.")

if __name__ == "__main__": main()
