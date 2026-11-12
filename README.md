# 四智能体证据驱动方法

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
