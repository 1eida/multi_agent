from pathlib import Path
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
