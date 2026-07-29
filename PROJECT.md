# RideWay 项目详解

RideWay 是一个评测 LLM Agent 在真实打车服务场景中处理多轮对话任务能力的 Benchmark。整个系统由三个 LLM 协同工作：**Agent**（被测）、**User Simulator**（模拟用户）、**Evaluator**（评判官）。

---

## 一、整体运行流程

```
1. 准备阶段
   Task JSON ──→ Registry ──→ Environment（DB + Tools）

2. 对话阶段（Orchestrator 调度）
   User Simulator ◄──→ Agent ◄──→ Environment（工具执行）
                           │
                     对话轨迹 (trajectory)

3. 评测阶段
   trajectory + task.instructions + rubrics ──→ Evaluator LLM ──→ RewardInfo
```

### 按代码执行顺序

| 步骤 | 代码位置 | 说明 |
|------|----------|------|
| 1. 启动 | `run.py:run_domain` | 解析 `RunConfig`，加载领域配置 |
| 2. 加载任务 | `run.py:get_tasks` | 从 `data/eff/domains/{domain}/tasks.json` 读取 Task 列表 |
| 3. 构建环境 | `registry.get_env_constructor` | 用 `task.environment` 初始化 DB 和 Tools |
| 4. 创建 Agent/User | `LLMAgent` / `UserSimulator` | Agent 获取工具列表，User 获取用户画像和指令 |
| 5. 对话模拟 | `orchestrator.py:Orchestrator.run` | Agent↔User 轮流交互，Environment 执行工具调用 |
| 6. 评测 | `evaluator.py:evaluate_simulation` | 对话结束后，Evaluator LLM 按 rubric 逐条打分 |
| 7. 保存结果 | `run.py:_save` | SimulationRun 追加写入 JSON 文件 |

### 终止条件（满足任一）

- `USER_STOP`：User 发出结束语
- `AGENT_STOP`：Agent 判断任务完成
- `MAX_STEPS`：达到最大步数（默认 300）
- `TOO_MANY_ERRORS`：工具调用错误过多

---

## 二、添加新 Domain

### 目录结构

```
src/eff/domains/{my_domain}/
├── __init__.py
├── data_model.py      # 领域实体 + DB 子类
├── tools.py           # Agent 可调用的工具集
├── tools_schema.py    # 工具的多语言描述（可选）
└── environment.py     # 环境工厂函数 + 任务加载函数

data/eff/domains/{my_domain}/
├── tasks.json         # 中文任务
└── tasks_en.json      # 英文任务（可选）
```

### 2.1 data_model.py

定义领域实体和 DB 子类。DB 基类已提供通用字段，只需添加领域特有字段：

```python
from eff.data_model.db import DB

class MyDomainDB(DB):
    # DB 基类已有：time, user_id, weather, location, orders, user_historical_behaviors
    stores: Dict[str, Store] = Field(description="门店字典，key 为门店 ID")
    # ... 其他领域特有字段
```

**规范**：
- 必须继承 `DB` 基类
- 所有实体用 Pydantic `BaseModel` 定义，字段加 `Field(description=...)`
- 有"商店"类实体继承 `StoreBaseModel`，有"商品"类实体继承 `ProductBaseModel`

### 2.2 tools.py

定义 Agent 可调用的工具，用 `@is_tool` 装饰器标记暴露给 Agent 的方法：

```python
from eff.tools.base import ToolKitBase, is_tool, ToolType

class MyDomainTools(ToolKitBase):
    def __init__(self, db: MyDomainDB):
        super().__init__(db)

    @is_tool(ToolType.READ)           # 只读：查询、搜索
    def search_items(self, keywords: List[str]) -> str:
        """搜索商品"""
        assert keywords, "关键词不能为空"
        # ... 逻辑 ...
        return "搜索结果"

    @is_tool(ToolType.WRITE)          # 写操作：创建、修改、取消
    def create_order(self, item_id: str) -> str:
        """创建订单"""
        return "订单创建成功"

    def _internal_helper(self):       # 下划线开头：不暴露给 Agent
        pass
```

**规范**：
- 必须继承 `ToolKitBase`
- 公开工具用 `@is_tool(ToolType.READ/WRITE/GENERIC)` 装饰
- 内部方法以 `_` 开头
- 返回值必须是 `str`
- 用 `assert` 做参数校验（失败会被包装为错误信息而非崩溃）

ToolKitBase 已内置的通用工具（无需重写）：`longitude_latitude_to_distance`、`weather`、`address_to_longitude_latitude`、`get_date_holiday_info`、`get_user_historical_behaviors`、`get_user_all_orders`、`get_nearby`

### 2.3 environment.py

提供两个工厂函数供 Registry 调用：

```python
def get_environment(db=None, language=None) -> Environment:
    if db is None:
        db = {}
    my_db = MyDomainDB.model_validate(db)
    tools = MyDomainTools(my_db)
    return Environment(
        domain_name="my_domain",
        policy=get_agent_policy(language),
        tools=tools,
    )

def get_tasks(language=None) -> list[Task]:
    task_path = get_task_file_path("my_domain", language)
    with open(task_path) as fp:
        tasks = json.load(fp)
    return [Task.model_validate(t) for t in tasks]
```

### 2.4 注册到 Registry

在 `src/eff/registry.py` 底部添加：

```python
from eff.domains.my_domain.environment import get_environment as my_domain_env
from eff.domains.my_domain.environment import get_tasks as my_domain_tasks

registry.register_domain(my_domain_env, "my_domain")
registry.register_tasks(my_domain_tasks, "my_domain")
```

### Checklist

```
□ src/eff/domains/{my_domain}/__init__.py
□ data_model.py  — 定义 DB 子类 + 领域实体
□ tools.py       — 定义工具集，@is_tool 标记
□ tools_schema.py — 多语言工具描述（可选）
□ environment.py — get_environment() + get_tasks()
□ data/eff/domains/{my_domain}/tasks.json
□ registry.py    — 注册 domain 和 tasks
```

---

## 三、Task JSON 格式

任务文件位于 `data/eff/domains/{domain}/tasks.json`，是一个 Task 数组。

### 字段总览

```json
{
    "id": "custom_015_shanghai_mcdonalds_rain_nosmoking",
    "domain": "ridehailing",
    "environment": { ... },
    "user_scenario": { ... },
    "instructions": "...",
    "evaluation_criteria": { ... },
    "message_history": []
}
```

### 各字段说明

**`id`**（必填）：任务唯一标识符，自定义，保证不重复。

**`domain`**（必填）：必须与 Registry 中注册的领域名一致。

**`environment`**（必填）：环境初始数据，传给 `get_environment(db=task.environment)` 初始化 DB。

```json
"environment": {
    "time": "2025-06-21 11:20:00",
    "user_id": "U000001",
    "orders": {},
    "weather": [{ "city": "昆明市", "category": "雨", "datetime": "2025-06-21", "temperature": [17, 25] }],
    "location": [{ "address": "云南省昆明市...", "longitude": 102.71, "latitude": 25.05 }],
    "user_historical_behaviors": { "用户id": "U000001", "职业": "医生" },
    "stores": { "S12345": { "name": "某餐厅", "products": [] } }
}
```

> DB 基类字段（`time`/`user_id`/`orders`/`weather`/`location`/`user_historical_behaviors`）是固定的；领域特有字段（如 `stores`）由 DB 子类决定。

**`user_scenario`**（必填）：发给 UserSimulator 的用户画像，内容完全自由定义。

```json
"user_scenario": {
    "user_profile": {
        "用户id": "U000001", "职业": "医生", "性别": "男",
        "饮食禁忌": "忌高嘌呤,忌油炸", "性格": "做事急躁"
    }
}
```

**`instructions`**（必填）：用户真实意图，同时发给 UserSimulator（扮演用户）和 Evaluator（评判是否完成）。

```
"外面正在下大雨，你决定点个清淡的米线到科室，不要油炸和高嘌呤的。你下午一点半有手术，需要午休一小时。"
```

**`evaluation_criteria`**（可选，为 `null` 时自动得 1.0 分）：

```json
"evaluation_criteria": {
    "expected_states": [
        {
            "required_orders": [{ "store_id": "S12345", "status": "processed", "products": [...] }],
            "optional_orders": [],
            "state_rubrics": [
                "下单的米线商品不含油炸配菜",
                "配送地址应为用户工作地址"
            ]
        }
    ],
    "overall_rubrics": [
        "Agent 应主动询问用户的口味偏好",
        "最终订单不应包含用户忌讳的食材"
    ]
}
```

**`message_history`**（可选）：预设的对话历史，通常为空数组 `[]`。

---

## 四、评测系统详解

### 4.1 数据流

```
task.instructions ──────────────────────────────┐
task.evaluation_criteria ──────────────────────┐│
                                               ││
Agent + User + Environment 对话 ──→ trajectory  ││
                                       │       ││
                                       ▼       ▼▼
                               TrajectoryEvaluator（LLM 评判官）
                                       │
                               RewardInfo {
                                 reward: 0.0 或 1.0,
                                 nl_rubrics: [{met, justification}, ...],
                                 rubric_score: 满足数/总数
                               }
```

### 4.2 四种评测模式

两个维度的正交组合：

|             | 有独立 rubric（逐条打分）        | 无独立 rubric（整体打分）         |
|------------ |-------------------------------|--------------------------------|
| **滑动窗口** | `trajectory`（默认）            | `trajectory_sliding_wo_rubric` |
| **完整轨迹** | `trajectory_full_traj_rubric`  | `trajectory_full_traj_wo_rubric` |

**有 rubric vs 无 rubric**：
- 有 rubric：Evaluator 逐条判断每个评分点 true/false，可解释性强，能定位具体失败环节
- 无 rubric：Evaluator 依据 instructions 给一个整体结论，写任务省事但可控性差

**滑动窗口 vs 完整轨迹**：

完整轨迹的问题：一次对话可能有几万 token，超出上下文限制；即使放得下，LLM 在长文本中注意力会分散（"Lost in the Middle"）；且无法追踪状态在对话过程中的变化。

滑动窗口的解决方案：
- 每个窗口默认 10 条消息，控制输入量恒定
- **rubric 状态跨窗口继承**：每个窗口看到当前对话 + 之前所有窗口的累积判断状态
- **支持状态回退**：Agent 前面做对但后来搞错，窗口可以将 `true` 翻转为 `false`
- **重叠 2 条**保证跨窗口边界的操作不会断裂

**选用建议**：

| 场景 | 推荐模式 |
|------|----------|
| 长对话（>10轮）+ 需要细粒度分析 | `trajectory`（默认）|
| 短对话（<10轮）+ 需要细粒度分析 | `trajectory_full_traj_rubric` |
| 长对话 + 只要整体结论 | `trajectory_sliding_wo_rubric` |
| 短对话 + 只要整体结论 | `trajectory_full_traj_wo_rubric` |

RideWay 默认使用 `trajectory`，因为典型场景对话较长且需要精确定位问题。

### 4.3 对已有对话直接打分（Re-evaluate）

如果对话轨迹已经存在，可跳过对话模拟阶段直接打分：

```bash
# CLI 方式
eff run \
  --re-evaluate-file data/simulations/xxx.json \
  --evaluator-llm deepseek-chat \
  --save-to re_eval_result
```

```python
# 代码方式
from eff.evaluator.evaluator import evaluate_simulation

for simulation in results.simulations:
    task = task_dict[simulation.task_id]
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type="trajectory",
        domain="ridehailing",
        llm_evaluator="deepseek-chat",
    )
```

如果数据不是 RideWay 格式，需要将对话组装为 `SimulationRun.messages`（`list[AssistantMessage | UserMessage | ToolMessage]`）和对应的 `Task`（含 `evaluation_criteria`）。
