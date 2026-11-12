"""Offline regression tests. No model server, network, or GPU is used."""

import json
import unittest
from types import SimpleNamespace

from multi_agent.ledger import EvidenceLedger


def test_ledger_bounds_and_provenance():
    ledger = EvidenceLedger()
    first = ledger.record("lookup", {"b": 2, "a": 1}, True, "x" * 100_000,
                          round=1, step=1)
    ledger.record("lookup", {"q": "bad"}, False, error="bad query", round=1, step=2)
    assert first["id"] == "e1"
    assert ledger.find("lookup", {"a": 1, "b": 2})["id"] == "e1"
    assert ledger.find("lookup", {"q": "bad"}, failed_only=True)["id"] == "e2"
    assert ledger.valid_ids(["e1", "e999", "e1", "e2"]) == ["e1", "e2"]
    view = ledger.render(max_obs_chars=200, max_total_chars=1000)
    assert len(view) <= 1000
    parsed = json.loads(view)
    assert parsed["observations"][0]["truncated"]
    assert parsed["observations"][0]["original_chars"] >= 100_000


class ScriptedModel:
    def __init__(self, scripts):
        self.scripts = {role: list(values) for role, values in scripts.items()}
        self.calls = []

    def ask(self, system, payload, *, role, json_mode=True):
        self.calls.append((role, system, payload))
        if not self.scripts.get(role):
            raise AssertionError(f"Unexpected {role} call")
        return self.scripts[role].pop(0)


class FakeTools:
    def __init__(self, result=None):
        self.calls = []
        self.result = result if result is not None else {"items": ["alpha", "noise"]}

    def catalog(self):
        return [{"name": "lookup", "description": "Look up records", "parameters": {"type": "object"}, "read_only": True}]

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        return self.result


def plan(update=None):
    return {"epistemic_update": update or {}, "goal": "Find the matching item",
            "output_contract": "Return the matching item only",
            "steps": [{"step_id": "1", "sub_goal": "Find records", "tool_name": "lookup", "purpose": "Check records"}]}


def plan_review(update=None):
    return {"epistemic_update": update or {}, "approved": True,
            "acceptance_predicates": ["Only matching items"], "execution_guardrails": [],
            "probe_instruction": "Check scope"}


def action(query="initial"):
    return {"action": "tool", "name": "lookup", "arguments": {"q": query}, "reason": "Inspect records"}


def final(answer="alpha", **kwargs):
    return {"action": "final", "answer": answer, "cited_evidence_ids": ["e1"],
            "unresolved_issues": [], **kwargs}


def review(verdict="accept", update=None):
    return {"verdict": verdict, "epistemic_update": update or {},
            "cited_evidence_ids": ["e1"], "diagnostic_task_for_planner": {
                "failure_mode": "noise", "competing_hypotheses": ["matching", "not matching"],
                "next_discriminative_check": "Check scope", "decision_rule": "Exclude nonmatches"}}


def test_small_model_normalization_and_provenance():
    from multi_agent.agents import Planner, PlanValidator, WorkerValidator, normalize_memory

    p = plan()
    p["goal"] = {"target": "alpha"}
    p["output_contract"] = ["one item"]
    normalized = Planner.normalize(p)
    assert json.loads(normalized["goal"]) == {"target": "alpha"}
    assert json.loads(normalized["output_contract"]) == ["one item"]
    assert PlanValidator.normalize({"approved": "false", "acceptance_predicates": "match"})["approved"] is False
    assert WorkerValidator.normalize({"verdict": "accept"})["verdict"] == "accept"
    update = {"verified_anchors": [{"claim": "x", "evidence_ids": ["e1", "e999"]},
                                    {"claim": "unsupported", "evidence_ids": ["e999"]}]}
    normalized_update = normalize_memory("planner", update, ["e1"])
    assert normalized_update["verified_anchors"] == [{"claim": "x", "evidence_ids": ["e1"]}]


def task():
    return SimpleNamespace(task_id="offline", question="Return the matching item", context={"scope": "test"})


def test_multiround_four_role_evolution():
    from multi_agent import Config, run
    scripts = {
        "planner": [plan(), plan({"verified_anchors": [{"claim": "alpha is a candidate", "evidence_ids": ["e1"]}],
                                  "falsified_paths": [{"claim": "noise is off target", "evidence_ids": ["e1"]}]})],
        "plan_validator": [plan_review(), plan_review({"learned_guardrails": [{"rule": "Check scope", "evidence_ids": ["e1"]}]})],
        "worker": [action(), final("alpha, noise", epistemic_update={"effective_tactics": [{"note": "inspect records", "evidence_ids": ["e1"]}]}), action("scope"), final()],
        "worker_validator": [review("revise", {"rejected_noise": [{"item": "noise", "reason": "scope", "evidence_ids": ["e1"]}],
                                                "diagnostic_probes": [{"probe": "scope", "status": "pending", "evidence_ids": ["e1"]}]}),
                             review("accept", {"verified_claims": [{"claim": "alpha", "evidence_ids": ["e1"]}]})],
    }
    model, tools = ScriptedModel(scripts), FakeTools()
    result = run(task(), model, tools, config=Config(max_rounds=2))
    assert result.status == "completed"
    assert result.answer == "alpha"
    assert result.artifacts["rounds_completed"] == 2
    snapshot = result.artifacts["role_memories_history"][-1]
    assert snapshot["planner"]["verified_anchors"]
    assert snapshot["plan_validator"]["learned_guardrails"]
    assert snapshot["worker"]["effective_tactics"]
    assert snapshot["worker_validator"]["rejected_noise"]
    second_plan = [p for r, _, p in model.calls if r == "planner"][1]
    assert second_plan["previous_work_review"]["diagnostic_task_for_planner"]["failure_mode"] == "noise"


def test_one_step_schema_repair():
    from multi_agent import Config, run
    model = ScriptedModel({"planner": [{"goal": "incomplete"}, plan()],
                           "plan_validator": [plan_review()], "worker": [action(), final()],
                           "worker_validator": [review()]})
    result = run(task(), model, FakeTools(), config=Config(max_rounds=1))
    assert result.status == "completed"
    assert sum(bool(x.get("repair")) for x in result.trace if x.get("event") == "model_call") == 1


def test_optional_structured_generation_preserves_role_schemas_repair_and_nested_answer():
    import jsonschema
    from multi_agent import Config, run
    from multi_agent.agents import Planner, PlanValidator, Worker, WorkerValidator

    for role in (Planner, PlanValidator, Worker, WorkerValidator):
        with unittest.TestCase().assertRaises(jsonschema.ValidationError):
            jsonschema.validate({"sql": "SELECT 1"}, role.response_schema)
    for role, value in ((Planner, plan()), (PlanValidator, plan_review()),
                        (Worker, action()), (Worker, final({"sql": "SELECT 1"})),
                        (WorkerValidator, review())):
        jsonschema.validate(value, role.response_schema)

    class CapableModel(ScriptedModel):
        def __init__(self, scripts):
            super().__init__(scripts)
            self.structured_roles = []

        def ask(self, *args, **kwargs):
            raise AssertionError("structured model must not call plain ask")

        def ask_structured(self, system, payload, *, role, response_schema):
            self.structured_roles.append(role)
            if role == "worker":
                assert response_schema == Worker.constrained_schema(FakeTools().catalog(),
                    allow_final=not bool(payload.get("empirical_gaps")))
            else:
                assert response_schema == {r.name: r.response_schema for r in
                                           (Planner, PlanValidator, Worker, WorkerValidator)}[role]
            return super().ask(system, payload, role=role)

    answer = {"sql": "SELECT 1"}
    model = CapableModel({"planner": [{"sql": "SELECT 1"}, plan()],
                          "plan_validator": [plan_review()], "worker": [action(), final(answer)],
                          "worker_validator": [review()]})
    result = run(task(), model, FakeTools(), config=Config(max_rounds=1))
    assert result.status == "completed"
    assert result.answer == answer
    assert model.structured_roles == ["planner", "planner", "plan_validator", "worker", "worker", "worker_validator"]
    assert sum(bool(event.get("repair")) for event in result.trace) == 1


def test_role_schemas_allow_documented_optional_fields_and_reject_unknown_outer_keys():
    import jsonschema
    from multi_agent.agents import Planner, PlanValidator, Worker, WorkerValidator

    examples = [(Planner, plan()), (PlanValidator, plan_review()),
                (Worker, action()), (Worker, final({"arbitrary": {"nested": [None, 1, "x"]}})),
                (WorkerValidator, review())]
    for role, value in examples:
        # Normalization emits the full documented optional outer envelope.
        normalized = role.normalize(value)
        jsonschema.validate(normalized, role.response_schema)
        with unittest.TestCase().assertRaises(jsonschema.ValidationError):
            jsonschema.validate({**normalized, "review": "unsupported outer alias"}, role.response_schema)
    # Optional citations remain optional; constrained syntax cannot invent
    # evidence, require acceptance, or bypass the host's grounding predicate.
    jsonschema.validate({"verdict": "revise"}, WorkerValidator.response_schema)
    jsonschema.validate({"approved": False}, PlanValidator.response_schema)
    jsonschema.validate({"verdict": "accept", "cited_evidence_ids": [],
                         "epistemic_update": {"arbitrary_memory": {"nested": [1, 2]}}},
                        WorkerValidator.response_schema)


def test_transport_value_error_gets_contextual_repair():
    from multi_agent import Config, run

    class TruncatedModel(ScriptedModel):
        def ask(self, system, payload, *, role, json_mode=True):
            if role == "worker_validator" and not any(r == role for r, _, _ in self.calls):
                self.calls.append((role, system, payload))
                raise ValueError("incomplete model response: length")
            return super().ask(system, payload, role=role, json_mode=json_mode)

    model = TruncatedModel({"planner": [plan()], "plan_validator": [plan_review()],
                            "worker": [action(), final()], "worker_validator": [review()]})
    result = run(task(), model, FakeTools(), config=Config(max_rounds=1))
    assert result.status == "completed"
    repaired = [p for r, _, p in model.calls if r == "worker_validator"][-1]
    assert repaired["candidate"]["answer"] == "alpha"
    assert repaired["repair_error"] == "incomplete model response: length"
    assert sum(bool(x.get("repair")) for x in result.trace if x.get("event") == "model_call") == 1


def test_fixed_round_preserves_accepted_answer_on_regression_and_budget():
    from multi_agent import Config, run
    model = ScriptedModel({"planner": [plan(), plan(), plan()],
                           "plan_validator": [plan_review(), plan_review(), plan_review()],
                           "worker": [action(), final(), final("unsupported addition"), action("different"), action("different")],
                           "worker_validator": [review(), review(), review()]})
    result = run(task(), model, FakeTools(), config=Config(max_rounds=3, allow_early_stop=False,
                                                         max_actions_per_round=1))
    assert result.status == "completed" and result.answer is not None
    assert result.artifacts["completion_mode"] == "strict_accept"


def test_ablation_isolation():
    for enabled, roles in [(False, ("planner", "plan_validator", "worker", "worker_validator")),
                           (True, ("worker",))]:
        _check_ablation(enabled, roles)


def _check_ablation(enabled, roles):
    from multi_agent import Config, run
    update = {"effective_tactics": [{"note": "lookup", "evidence_ids": ["e1"]}]}
    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                           "worker": [action(), final(epistemic_update=update)],
                           "worker_validator": [review()]})
    result = run(task(), model, FakeTools(), config=Config(max_rounds=1, enable_coevolution=enabled, evolve_roles=roles))
    assert result.status == "completed"
    state = result.artifacts["role_memories_history"][0]
    assert bool(state["worker"]["effective_tactics"]) == enabled
    assert not state["planner"]["verified_anchors"]
    assert not state["plan_validator"]["learned_guardrails"]
    assert not state["worker_validator"]["verified_claims"]


def test_budget_exhaustion_and_synthesis():
    from multi_agent import Config, run
    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                           "worker": [action(), final()], "worker_validator": [review()]})
    result = run(task(), model, FakeTools(), config=Config(max_rounds=1, max_tool_calls=0))
    assert result.status == "budget_exhausted"
    assert any(payload.get("tools_disabled") for role, _, payload in model.calls if role == "worker")
    model = ScriptedModel({"planner": [plan()]})
    result = run(task(), model, FakeTools(), config=Config(max_rounds=1, max_model_calls=1))
    assert result.status == "budget_exhausted"
    assert result.artifacts["model_calls"] == 1


def test_oversized_tool_output_does_not_bloat_prompts():
    from multi_agent import Config, run
    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                           "worker": [action(), final()], "worker_validator": [review()]})
    result = run(task(), model, FakeTools("x" * 100_000),
                 config=Config(max_rounds=1, max_obs_chars=300, max_total_evidence_chars=1500))
    assert result.status == "completed"
    assert all(len(system) + len(json.dumps(payload)) < 20_000 for _, system, payload in model.calls)
    assert any('"truncated":true' in json.dumps(payload.get("evidence", ""), separators=(",", ":")).replace('\\"', '"')
               for _, _, payload in model.calls)


def test_large_structured_tasks_and_tool_schemas_survive_all_roles_and_repair():
    from multi_agent import Config, run

    schema = [{"table": f"sample_table_{i}",
               "ddl": f"CREATE TABLE sample_table_{i} (subject_key INTEGER, "
                      + ", ".join(f"measurement_{j} REAL" for j in range(20)) + ");"}
              for i in range(15)]
    contract = 'Return exactly a JSON object {"sql":"the SQL query"}.'
    catalog = [{"name": name, "description": "Registered operation " * 30,
                "parameters": {"type": "object", "properties": {
                    "q": {"type": "string", "description": "The public query"}},
                    "required": ["q"], "additionalProperties": False}, "read_only": True}
               for name in ["lookup"] + [f"operation_{i}" for i in range(7)]]

    class ManyTools(FakeTools):
        def catalog(self):
            return catalog

    contexts = [
        {"schema": schema, "fixture_clock": "2100-12-31 23:59:00", "response_format": contract},
        {"case_context": "Public subject context " * 100,
         "function_documentation": "Public API documentation " * 400,
         "response_format": "Return the final answer as a JSON list."},
    ]
    for context in contexts:
        public_task = SimpleNamespace(task_id="long-public-task",
                                      question="Inspect the provided records. " * 100 + contract,
                                      context=context)
        model = ScriptedModel({"planner": [{"goal": "incomplete"}, plan()],
                               "plan_validator": [plan_review()], "worker": [action(), final()],
                               "worker_validator": [review()]})
        result = run(public_task, model, ManyTools(), config=Config(max_rounds=1))
        assert result.status == "completed"
        assert any(event.get("repair") for event in result.trace)
        for _, _, payload in model.calls:
            assert payload["task"]["task_id"] == public_task.task_id
            assert payload["task"]["question"] == public_task.question
            assert payload["task"]["context"] == context
            assert payload["task"]["context"]["response_format"] == context["response_format"]
            assert payload["catalog"] == catalog
            assert len(json.dumps(payload, ensure_ascii=True)) <= 48000


def test_global_prompt_pressure_keeps_task_contract_and_complete_catalog():
    from multi_agent import Config, run

    catalog = [{"name": name, "description": 'Long operation documentation "\\ ' * 100,
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                               "required": ["q"]}, "read_only": True}
               for name in ["lookup"] + [f"operation_{i}" for i in range(9)]]

    class LargeTools(FakeTools):
        def catalog(self):
            return catalog

    response_format = 'Return a JSON object with the sql field, without prose.'
    context = {"schema": [{"table": "sample_events", "ddl": "subject_key INTEGER, " * 700}],
               "response_format": response_format}
    question = "Retrieve the requested sample information. " * 150 + response_format
    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                           "worker": [action(str(i)) for i in range(6)] + [final()],
                           "worker_validator": [review()]})
    result = run(SimpleNamespace(task_id="pressure", question=question, context=context),
                 model, LargeTools('\"\\医' * 20000),
                 config=Config(max_rounds=1, max_obs_chars=4000, max_total_evidence_chars=24000))
    assert result.status == "completed"
    for _, _, payload in model.calls:
        assert len(json.dumps(payload, ensure_ascii=True)) <= 48000
        assert payload["task"]["question"] == question
        assert payload["task"]["context"] == context
        assert [tool["name"] for tool in payload["catalog"]] == [tool["name"] for tool in catalog]
        assert [tool["parameters"] for tool in payload["catalog"]] == [tool["parameters"] for tool in catalog]
    final_evidence = [payload["evidence"] for role, _, payload in model.calls if role == "worker"][-1]
    assert len(final_evidence) < 24000


def test_oversized_task_text_keeps_question_tail_contract_and_schema_structure():
    from multi_agent import Config, run

    contract = 'Return exactly {"sql":"the successful SELECT"}.'
    schema = [{"table": f"table_{i}", "ddl": f"CREATE TABLE table_{i}(subject_key INTEGER);"}
              for i in range(10)]
    public_task = SimpleNamespace(task_id="long-text", question=(
        "Find the requested sample fact. " + "Public question detail 医\\\" " * 2000 + contract),
        context={"documentation": "Public documentation 医\\\" " * 4000,
                 "schema": schema, "response_format": contract})
    model = ScriptedModel({"planner": [plan()]})
    run(public_task, model, FakeTools(), config=Config(max_rounds=1, max_model_calls=1))
    payload = model.calls[0][2]
    projected = payload["task"]
    assert isinstance(projected["question"], str)
    assert projected["question"].startswith("Find the requested sample fact.")
    assert projected["question"].endswith(contract)
    assert len(json.dumps(projected["question"], ensure_ascii=True)) <= 8000
    assert projected["context"]["response_format"] == contract
    assert projected["context"]["schema"] == schema
    assert len(json.dumps(projected["context"], ensure_ascii=True)) <= 16000
    assert "[truncated; original_chars=" in projected["context"]["documentation"]
    assert public_task.context["documentation"].count("Public documentation") == 4000


def test_failed_duplicates_are_reused_and_lookup_is_windowed():
    from multi_agent import Config, run

    class FailingTools(FakeTools):
        def call(self, name, arguments):
            self.calls.append((name, arguments))
            raise ValueError("bad query")

    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                           "worker": [action(), action(),
                                      {"action": "tool", "name": "ledger_lookup", "arguments": {"id": "e1", "offset": 0}},
                                      final()], "worker_validator": [review("revise")]})
    tools = FailingTools()
    result = run(task(), model, tools, config=Config(max_rounds=1))
    assert len(tools.calls) == 1
    assert result.artifacts["ledger_summary"]["count"] == 1
    assert any(event["event"] == "tool_reused" for event in result.trace)
    assert any(event["event"] == "ledger_lookup" for event in result.trace)


def test_new_task_has_empty_memories_and_ledger():
    from multi_agent import Config, run

    for _ in range(2):
        model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                               "worker": [action(), final()], "worker_validator": [review()]})
        result = run(task(), model, FakeTools(), config=Config(max_rounds=1))
        assert result.artifacts["ledger_summary"]["observations"][0]["id"] == "e1"
        assert not any(model.calls[0][2]["memory"].values())


def test_zero_tool_shortcut_gets_bounded_in_round_nudge():
    from multi_agent import Config, run
    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                           "worker": [final(), action(), final()], "worker_validator": [review()]})
    tools = FakeTools()
    result = run(task(), model, tools, config=Config(max_rounds=1))
    assert result.status == "completed" and len(tools.calls) == 1
    assert any(e["event"] == "premature_final_blocked" for e in result.trace)
    worker_packets = [p for r, _, p in model.calls if r == "worker"]
    assert worker_packets[1]["last_observation"]["final_rejected"]


def test_empirical_revision_requires_new_probe_and_exposes_host_rejection():
    from multi_agent import Config, run
    model = ScriptedModel({"planner": [plan(), plan()], "plan_validator": [plan_review(), plan_review()],
                           "worker": [action(), final(), final(), action("alternative"), final()],
                           "worker_validator": [review("revise"), review()]})
    tools = FakeTools()
    result = run(task(), model, tools, config=Config(max_rounds=2))
    assert result.status == "completed" and len(tools.calls) == 1
    assert result.answer == "alpha"
    assert result.artifacts["completion_mode"] in {"grounded_round_commit", "strict_accept"}


def test_format_only_revision_can_reuse_verified_evidence():
    from multi_agent import Config, run
    correction = {**review("revise"), "evidence_complete": True,
                  "format_fix_instruction": "Return a scalar array."}
    model = ScriptedModel({"planner": [plan(), plan()], "plan_validator": [plan_review(), plan_review()],
                           "worker": [action(), final({"value": "alpha"}), final(["alpha"])],
                           "worker_validator": [correction, review()]})
    tools = FakeTools()
    result = run(task(), model, tools, config=Config(max_rounds=2))
    assert result.status == "completed" and result.answer == ["alpha"] and len(tools.calls) == 1


def test_failed_tool_repairs_within_round_and_failed_citations_cannot_ground():
    from multi_agent import Config, run

    class RepairTools(FakeTools):
        def call(self, name, arguments):
            self.calls.append((name, arguments))
            if arguments["q"] == "bad":
                raise ValueError("Invalid URL: encode query parameters")
            return {"items": ["alpha"]}

    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                           "worker": [action("bad"), final(), action("fixed"),
                                      final(cited_evidence_ids=["e2"])],
                           "worker_validator": [{**review(), "cited_evidence_ids": ["e2"]}]})
    tools = RepairTools()
    result = run(task(), model, tools, config=Config(max_rounds=1))
    assert result.status == "completed" and len(tools.calls) == 2
    packets = [p for r, _, p in model.calls if r == "worker"]
    assert "Repair" in packets[1]["last_observation"]["instruction"]
    # A permissive model audit cannot turn a failed call into supporting evidence.
    bad = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                         "worker": [action("bad"), final()], "worker_validator": [review()]})
    rejected = run(task(), bad, RepairTools(), config=Config(max_rounds=1, max_actions_per_round=1))
    assert rejected.status == "budget_exhausted" and rejected.answer is None
    assert any(e["event"] == "empirical_gate" for e in rejected.trace)


def test_host_empirical_gate_feedback_survives_rubber_stamp():
    from multi_agent import Config, run
    model = ScriptedModel({"planner": [plan(), plan()], "plan_validator": [plan_review(), plan_review()],
                           "worker": [final(), final(), action(), final()],
                           "worker_validator": [review(), review()]})
    result = run(task(), model, FakeTools(), config=Config(max_rounds=2, max_actions_per_round=1))
    assert result.status == "completed"
    second = [p for r, _, p in model.calls if r == "planner"][1]
    assert second["previous_work_review"]["verdict"] == "revise"
    assert second["previous_work_review"]["unresolved_issues"]


def test_preview_shortcut_requires_full_inspection_and_keeps_scalar_answer():
    from multi_agent import Config, run

    class RetrievalTools(FakeTools):
        def catalog(self):
            return [{"name": "find", "description": "Find document snippets", "parameters": {}, "read_only": True},
                    {"name": "inspect", "description": "Retrieve complete document text", "parameters": {}, "read_only": True}]
        def call(self, name, arguments):
            self.calls.append((name, arguments))
            return {"documents": [{"doc_id": "d1", "snippet": "A might relate to B"}]} if name == "find" else {"text": "A relates to B via C."}

    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                           "worker": [{"action": "tool", "name": "find", "arguments": {}}, final(),
                                      {"action": "tool", "name": "inspect", "arguments": {"doc_id": "d1"}},
                                      final([123], cited_evidence_ids=["e2"])],
                           "worker_validator": [{**review(), "cited_evidence_ids": ["e2"]}]})
    tools = RetrievalTools()
    result = run(task(), model, tools, config=Config(max_rounds=1))
    assert result.status == "completed" and result.answer == [123]
    assert [name for name, _ in tools.calls] == ["find", "inspect"]
    assert any("full content" in gap for e in result.trace for gap in e.get("gaps", []))


def test_empty_action_answer_and_confirmed_empty_query_are_valid():
    from multi_agent import Config, run
    for observation in ({"acknowledged": True}, {"rows": []}):
        model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                               "worker": [action(), final([])], "worker_validator": [review()]})
        result = run(task(), model, FakeTools(observation), config=Config(max_rounds=1))
        assert result.status == "completed" and result.answer == []
    # A second discriminative query can establish that a prior empty result was real.
    model = ScriptedModel({"planner": [plan(), plan()], "plan_validator": [plan_review(), plan_review()],
                           "worker": [action(), final([]), action("confirm absence"), final([])],
                           "worker_validator": [review("revise"), review()]})
    result = run(task(), model, FakeTools({"rows": []}), config=Config(max_rounds=2))
    assert result.status == "completed" and result.answer == []


def test_public_context_can_ground_tasks_without_tools():
    from multi_agent import Config, run

    class NoTools(FakeTools):
        def catalog(self):
            return []
    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                           "worker": [final(["provided"], cited_evidence_ids=["public:task"])],
                           "worker_validator": [{**review(), "cited_evidence_ids": ["public:task"]}]})
    result = run(task(), model, NoTools(), config=Config(max_rounds=1))
    assert result.status == "completed" and not result.artifacts["tool_calls"]


def test_forced_synthesis_and_audit_citation_failures_get_format_repair_feedback():
    from multi_agent import Config, run
    for candidate_ids, audit_ids in (([], ["e1"]), (["e1"], [])):
        model = ScriptedModel({"planner": [plan(), plan()], "plan_validator": [plan_review(), plan_review()],
                               "worker": [action(), final(cited_evidence_ids=candidate_ids), final()],
                               "worker_validator": [{**review(), "cited_evidence_ids": audit_ids}, review()]})
        tools = FakeTools()
        result = run(task(), model, tools,
                     config=Config(max_rounds=2, max_actions_per_round=1, max_tool_calls=1))
        assert result.status == "completed" and len(tools.calls) == 1
        assert result.artifacts["rounds_completed"] == 1
        assert result.answer is not None


def test_empirical_revise_at_tool_cap_can_resolve_from_existing_successful_evidence():
    from multi_agent import Config, run
    model = ScriptedModel({"planner": [plan(), plan()], "plan_validator": [plan_review(), plan_review()],
                           "worker": [action(), final(), final()],
                           "worker_validator": [review("revise"), review()]})
    result = run(task(), model, FakeTools(), config=Config(max_rounds=2, max_tool_calls=1))
    assert result.status == "completed" and result.artifacts["tool_calls"] == 1
    packet = [p for r, _, p in model.calls if r == "worker"][-1]
    assert not packet["verification_requirements"]["new_probe_required"]


def test_structured_worker_schema_requires_real_tool_then_detail_then_preserves_final():
    import jsonschema
    from multi_agent import Config, run
    from multi_agent.agents import Worker

    class Tools(FakeTools):
        def catalog(self):
            return [{"name": "locate", "description": "Find snippets", "parameters": {
                        "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}, "read_only": True},
                    {"name": "open", "description": "Read full document content", "parameters": {
                        "type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}, "read_only": True}]
        def call(self, name, arguments):
            self.calls.append((name, arguments))
            return {"snippet": "A leads to B"} if name == "locate" else {"text": "B supports C"}

    class Structured(ScriptedModel):
        def ask_structured(self, system, payload, *, role, response_schema):
            if role == "worker":
                if not self.calls or not any(r == "worker" for r, _, _ in self.calls):
                    with unittest.TestCase().assertRaises(jsonschema.ValidationError):
                        jsonschema.validate(final(), response_schema)
                    with unittest.TestCase().assertRaises(jsonschema.ValidationError):
                        jsonschema.validate({"action": "tool", "name": "locate", "arguments": {}}, response_schema)
                elif payload.get("empirical_gaps"):
                    with unittest.TestCase().assertRaises(jsonschema.ValidationError):
                        jsonschema.validate({"action": "tool", "name": "locate", "arguments": {"q": "again"}}, response_schema)
                    assert response_schema["properties"]["name"]["enum"] == ["open"]
            value = super().ask(system, payload, role=role)
            jsonschema.validate(value, response_schema)
            return value

    model = Structured({"planner": [plan()], "plan_validator": [plan_review()],
                        "worker": [{"action": "tool", "name": "locate", "arguments": {"q": "A"}},
                                   {"action": "tool", "name": "open", "arguments": {"id": "B"}},
                                   final(["C"], cited_evidence_ids=["e2"])],
                        "worker_validator": [{**review(), "cited_evidence_ids": ["e2"]}]})
    result = run(task(), model, Tools(), config=Config(max_rounds=1))
    assert result.status == "completed" and result.answer == ["C"]
    jsonschema.validate(final([]), Worker.constrained_schema([], tools_disabled=True))
    with unittest.TestCase().assertRaises(jsonschema.ValidationError):
        jsonschema.validate(action(), Worker.constrained_schema([], tools_disabled=True))


def test_required_plan_checks_cannot_be_claimed_after_only_existence_probe():
    from multi_agent import Config, run
    import jsonschema

    class VerificationTools(FakeTools):
        def catalog(self):
            return [{"name": name, "description": "Verify one investigation stage", "parameters": {
                "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}, "read_only": True}
                for name in ("exists", "attribute", "external")]

    investigation = {**plan(), "steps": [{"sub_goal": name, "tool_name": name, "required": True}
                                        for name in ("exists", "attribute", "external")]}
    actions = [{"action": "tool", "name": name, "arguments": {"q": "entity"}}
               for name in ("exists", "attribute", "external")]
    model = ScriptedModel({"planner": [investigation], "plan_validator": [plan_review()],
                           "worker": [*actions, final()],
                           "worker_validator": [review()]})
    tools = VerificationTools()
    result = run(task(), model, tools, config=Config(max_rounds=1))
    assert result.status == "completed"
    assert [name for name, _ in tools.calls] == ["exists", "attribute", "external"]
    packets = [p for r, _, p in model.calls if r == "worker"]
    assert packets[0]["required_tool_names"] == ["exists", "attribute", "external"]
    assert packets[1]["required_tool_names"] == ["attribute", "external"]
    assert packets[2]["required_tool_names"] == ["external"]
    # At the final round the host releases an uncovered read-only check but
    # records it as unresolved and cannot accept it strictly.
    shortcut = ScriptedModel({"planner": [investigation], "plan_validator": [plan_review()],
                              "worker": [actions[0], final()], "worker_validator": [review()]})
    rejected = run(task(), shortcut, VerificationTools(),
                   config=Config(max_rounds=1, max_actions_per_round=1))
    assert rejected.status == "completed"
    assert rejected.answer is not None
    assert rejected.artifacts["completion_mode"] == "grounded_round_commit"


def test_tool_policy_tokenizes_mutation_names_without_read_false_positives():
    from multi_agent.workflow import _tool_policy
    for name in ("WRITE_SAMPLE", "submit_order", "createRecord", "write"):
        assert _tool_policy({"name": name, "description": "", "mutates": False}) == "mutating"
    for name in ("search", "read", "query_record_presence", "run_query", "read_report", "place_lookup"):
        assert _tool_policy({"name": name, "description": "", "read_only": True}) == "read_only"


def test_changed_terminal_failures_release_required_read_gate_without_absence_claim():
    from multi_agent import Config, run

    class Failing(FakeTools):
        def catalog(self):
            return [{"name": "lookup", "description": "Read records", "parameters": {}, "read_only": True}]
        def call(self, name, arguments):
            self.calls.append((name, arguments))
            raise RuntimeError("HTTP 404 Not Found")

    required = {**plan(), "steps": [{"sub_goal": "check", "tool_name": "lookup", "required": True}]}
    model = ScriptedModel({"planner": [required], "plan_validator": [plan_review()],
        "worker": [{"action": "tool", "name": "lookup", "arguments": {"q": "first"}},
                   {"action": "tool", "name": "lookup", "arguments": {"q": "repaired"}},
                   final("unknown", cited_evidence_ids=["e2"], unresolved_issues=["Lookup failed"])],
        "worker_validator": [review("revise")]})
    result = run(task(), model, Failing(), config=Config(max_rounds=3))
    assert len(result.artifacts["ledger_summary"]["observations"]) == 2
    assert result.artifacts["best_grounded"]["answer"] == "unknown"
    assert result.artifacts["completion_mode"] == "grounded_round_commit"
    assert "absence" not in str(result.answer).lower()
    assert result.artifacts["coverage_ids"] == ["e1", "e2"]


def test_mutating_only_catalog_is_never_host_forced():
    from multi_agent import Config, run

    class Writes(FakeTools):
        def catalog(self):
            return [{"name": "WRITE_SAMPLE", "description": "Write a record", "parameters": {}, "mutates": False}]
    model = ScriptedModel({"planner": [{**plan(), "steps": [{"sub_goal": "conditional", "tool_name": "WRITE_SAMPLE", "required": True}]}],
        "plan_validator": [plan_review()], "worker": [final("ok")], "worker_validator": [review()]})
    result = run(task(), model, Writes(), config=Config(max_rounds=1))
    packet = next(p for role, _, p in model.calls if role == "worker")
    assert packet["required_tool_names"] is None and packet["force_empirical_tool"] is False
    assert result.artifacts["tool_policy"] == {"WRITE_SAMPLE": "mutating"}
    assert result.artifacts["tool_calls"] == 0


def test_reserved_finalization_keeps_successful_read_answer_and_shape():
    from multi_agent import Config, run
    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
        "worker": [action(), final(["alpha"], cited_evidence_ids=["e1"])],
        "worker_validator": [review()]})
    result = run(task(), model, FakeTools({"rows": ["alpha"]}),
                 config=Config(max_rounds=1, max_model_calls=4))
    assert result.status == "completed" and result.answer == ["alpha"]
    assert result.artifacts["completion_mode"] == "grounded_round_commit"
    assert result.artifacts["tool_calls"] == 1


def test_search_and_read_required_checks_both_run_once():
    from multi_agent import Config, run
    class SearchRead(FakeTools):
        def catalog(self):
            return [{"name": n, "description": "Read records", "parameters": {}, "read_only": True}
                    for n in ("search", "read")]
    required = {**plan(), "steps": [{"sub_goal": n, "tool_name": n, "required": True}
                                    for n in ("search", "read")]}
    model = ScriptedModel({"planner": [required], "plan_validator": [plan_review()],
        "worker": [{"action": "tool", "name": "search", "arguments": {"q": "x"}},
                   {"action": "tool", "name": "read", "arguments": {"id": "x"}},
                   final("answer", cited_evidence_ids=["e1", "e2"])],
        "worker_validator": [review()]})
    tools = SearchRead()
    result = run(task(), model, tools, config=Config(max_rounds=1))
    assert [name for name, _ in tools.calls] == ["search", "read"]
    assert result.status == "completed"


def test_final_round_keeps_successful_empty_but_generic_error_is_not_absence():
    from multi_agent import Config, run

    class Lookup(FakeTools):
        def __init__(self, *, fail=False):
            super().__init__([])
            self.fail = fail
        def call(self, name, arguments):
            self.calls.append((name, arguments))
            if self.fail:
                raise RuntimeError("temporary service problem")
            return []

    for tools, answer, expected in ((Lookup(), [], "grounded_round_commit"),
                                    (Lookup(fail=True), "uncertain", None)):
        model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
            "worker": [action(), final(answer, cited_evidence_ids=["e1"],
                                       unresolved_issues=["The read failed"] if expected is None else [])],
            "worker_validator": [review("revise")]})
        result = run(task(), model, tools, config=Config(max_rounds=1, max_actions_per_round=1,
                                                        max_tool_calls=1))
        if expected:
            assert result.answer == [] and result.artifacts["completion_mode"] == expected
        else:
            assert result.status == "budget_exhausted" and result.answer is None


def test_explicit_optional_plan_probe_is_not_required_for_acceptance():
    from multi_agent import Config, run
    investigation = {**plan(), "steps": [*plan()["steps"],
        {"sub_goal": "Only if needed, run another check", "tool_name": "optional", "required": False}]}

    class OptionalTools(FakeTools):
        def catalog(self):
            return super().catalog() + [{"name": "optional", "description": "Conditional check", "parameters": {}, "read_only": True}]
    model = ScriptedModel({"planner": [investigation], "plan_validator": [plan_review()],
                           "worker": [action(), final()], "worker_validator": [review()]})
    result = run(task(), model, OptionalTools(), config=Config(max_rounds=1))
    assert result.status == "completed" and result.artifacts["tool_calls"] == 1


def test_planner_step_schema_rejects_unbounded_undeclared_property_names():
    import jsonschema
    from multi_agent.agents import Planner
    valid = {**plan(), "steps": [{**plan()["steps"][0], "required": True}]}
    jsonschema.validate(valid, Planner.response_schema)
    for key in ("unexpected", "purpose' "+"\r\n " * 1000):
        invalid = {**valid, "steps": [{**valid["steps"][0], key: "unbounded name"}]}
        with unittest.TestCase().assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid, Planner.response_schema)


def test_plan_validator_schema_bounds_reports_and_memory_without_changing_plain_normalization():
    import copy
    import jsonschema
    from multi_agent.agents import Planner, PlanValidator

    valid = {"approved": True, "acceptance_predicates": ["Verify the requested scope"],
             "execution_guardrails": ["Use registered tools"], "execution_warnings": [],
             "must_check": [], "execution_brief": "Investigate before deciding",
             "probe_instruction": "Discriminate competing explanations",
             "repaired_steps": copy.deepcopy(plan()["steps"]),
             "epistemic_update": {"learned_guardrails": [{
                 "text": "Use a discriminative observation", "claim": "A source supports the check",
                 "rule": "Inspect observations", "predicate": "Verified",
                 "reason": "Observed evidence", "evidence_ids": ["e1"]}],
                 "acceptance_predicates": ["Answer matches the requested scope"]}}
    schema = PlanValidator.response_schema
    jsonschema.validate(valid, schema)
    jsonschema.validate({"approved": False, "epistemic_update": {}, "repaired_steps": []}, schema)
    assert set(schema["properties"]["repaired_steps"]["items"]["properties"]) == set(Planner.response_schema["properties"]["steps"]["items"]["properties"])
    assert "maxLength" not in Planner.response_schema["properties"]["steps"]["items"]["properties"]["sub_goal"]
    invalid = []
    for field in ("acceptance_predicates", "execution_guardrails", "execution_warnings", "must_check"):
        invalid.extend([{**valid, field: ["check"] * 7}, {**valid, field: ["x" * 601]},
                        {**valid, field: [{"unexpected": "open-ended object"}]}])
    invalid.extend([{**valid, "execution_brief": "x" * 1201},
                    {**valid, "probe_instruction": "x" * 601},
                    {**valid, "repaired_steps": valid["repaired_steps"] * 6},
                    {**valid, "repaired_steps": [{"sub_goal": "x" * 601}]}])
    bad_key = "unknown" + "\r\n " * 1000
    invalid.extend([{**valid, bad_key: "unbounded property"},
                    {**valid, "epistemic_update": {bad_key: []}},
                    {**valid, "epistemic_update": {"learned_guardrails": [{bad_key: "unbounded"}]}},
                    {**valid, "repaired_steps": [{**valid["repaired_steps"][0], bad_key: "unbounded"}]},
                    {**valid, "epistemic_update": {"learned_guardrails": ["rule"] * 7}},
                    {**valid, "epistemic_update": {"learned_guardrails": [{"text": "x" * 601}]}},
                    {**valid, "epistemic_update": {"learned_guardrails": [{"evidence_ids": ["e1"] * 17}]}},
                    {**valid, "epistemic_update": {"learned_guardrails": [{"evidence_ids": ["e" * 65]}]}}])
    for value in invalid:
        with unittest.TestCase().assertRaises(jsonschema.ValidationError):
            jsonschema.validate(value, schema)
    # Plain scripted/third-party providers retain their prior normalization
    # contract; this change constrains structured generation rather than clipping.
    legacy = {"approved": "yes", "execution_guardrails": [{"rule": "legacy provider"}]}
    assert PlanValidator.normalize(legacy)["execution_guardrails"] == legacy["execution_guardrails"]


def test_public_premises_can_ground_with_tools_but_cannot_waive_empirical_execution():
    from multi_agent import Config, run
    public_task = SimpleNamespace(task_id="given", question="Inspect records and report the supplied scope", context={"scope": "test"})
    grounded = final("test", cited_evidence_ids=["e1"])
    audit = {**review(), "cited_evidence_ids": ["e1"]}
    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                           "worker": [grounded, action(), grounded], "worker_validator": [audit]})
    tools = FakeTools()
    result = run(public_task, model, tools, config=Config(max_rounds=1))
    assert result.status == "completed" and result.answer == "test" and len(tools.calls) == 1
    assert any(event["event"] == "premature_final_blocked" for event in result.trace)


def test_worker_validator_reason_cannot_repeat_until_output_token_limit():
    import jsonschema
    from multi_agent.agents import WorkerValidator
    jsonschema.validate({**review("revise"), "reason": "Tool failed; repair the public arguments before drawing a conclusion."},
                        WorkerValidator.response_schema)
    with unittest.TestCase().assertRaises(jsonschema.ValidationError):
        jsonschema.validate({**review("revise"), "reason": "The date parameter may be invalid. " * 1000},
                            WorkerValidator.response_schema)


def test_tool_state_rolls_back_between_rejected_rounds_and_commits_choice():
    from multi_agent import Config, run

    class StatefulTools(FakeTools):
        def __init__(self):
            super().__init__({"ok": True})
            self.state = {"writes": 0}
            self.restores = []

        def catalog(self):
            return [{"name": "write", "description": "Apply a state change", "parameters": {"type": "object"},
                     "mutates": False},
                    {"name": "read", "description": "Read current state", "parameters": {"type": "object"},
                     "read_only": True}]

        def call(self, name, arguments):
            self.calls.append((name, arguments))
            if name == "write":
                self.state["writes"] += 1
                return {"writes": self.state["writes"]}
            return {"writes": self.state["writes"]}

        def snapshot(self):
            return dict(self.state)

        def restore(self, snapshot):
            self.restores.append(dict(snapshot))
            self.state = dict(snapshot)

    model = ScriptedModel({"planner": [plan(), plan()], "plan_validator": [plan_review(), plan_review()],
                           "worker": [
                               {"action": "tool", "name": "write", "arguments": {}},
                               final("tentative"),
                               {"action": "tool", "name": "write", "arguments": {}},
                               final("committed")],
                           "worker_validator": [review("revise"), review("accept")]})
    tools = StatefulTools()
    result = run(task(), model, tools, config=Config(max_rounds=2, allow_early_stop=False))
    assert result.status == "completed" and result.answer == "committed"
    assert tools.state == {"writes": 1}
    assert {"writes": 0} in tools.restores
    assert result.artifacts["completion_mode"] == "strict_accept"


def test_final_round_commits_empirical_candidate_after_validator_revision():
    from multi_agent import Config, run
    model = ScriptedModel({"planner": [plan()], "plan_validator": [plan_review()],
                           "worker": [action(), final("best")],
                           "worker_validator": [review("revise")]})
    result = run(task(), model, FakeTools(), config=Config(max_rounds=1))
    assert result.status == "completed"
    assert result.answer == "best"
    assert result.artifacts["completion_mode"] == "grounded_round_commit"


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value, description=name)
                              for name, value in globals().items()
                              if name.startswith("test_") and callable(value))
