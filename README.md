# 设备异常推理 API

按设备型号提交振动、电流测值，按需装载固定权重的逻辑回归模型并返回异常分数，用于安排巡检。
三套模型、可配置推理延迟与字节预算通过 Docker Compose 提供，Swagger UI 可直接调用。
更新权重前可通过 `/v1/compare` 用同一段测值对照独立候选目录中的候选权重，确认巡检结论是否会改变；
正式 `/v1/infer` 始终只使用现行权重。

## 快速开始

```bash
docker compose up --build -d
curl -s localhost:8000/v1/models
open http://localhost:8000/docs          # Swagger UI
```

`POST /v1/infer` 示例：

```bash
curl -s localhost:8000/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"pump","vibration":[2.1,-1.9,3.4,-3.0],"current":[12.4,12.8,11.9,12.6]}'
```

```json
{
  "model_id": "pump",
  "model_name": "离心泵",
  "model_version": "1.0.0",
  "anomaly_score": 0.0312,
  "is_anomaly": false,
  "decision_threshold": 0.5,
  "features": {"vibration_mean": 0.15, "...": 0.0}
}
```

更新权重前，用同一段测值对照现行权重与候选权重（`POST /v1/compare`）：

```bash
curl -s localhost:8000/v1/compare \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"pump","candidate_version":"1.1.0-rc1",
       "vibration":[2.1,-1.9,3.4,-3.0],"current":[12.4,12.8,11.9,12.6]}'
```

```json
{
  "model_id": "pump",
  "model_name": "离心泵",
  "current":   {"model_version": "1.0.0", "anomaly_score": 0.0312,
                "is_anomaly": false, "decision_threshold": 0.5},
  "candidate": {"model_version": "1.1.0-rc1", "anomaly_score": 0.0487,
                "is_anomaly": false, "decision_threshold": 0.5},
  "score_difference": 0.0175,
  "decision_changed": false,
  "features": {"vibration_mean": 0.15, "...": 0.0}
}
```

`score_difference` 为候选分数减现行分数；`decision_changed` 直接表明巡检结论（是否异常）是否会改变。
候选权重只从独立候选目录读取，`/v1/infer` 始终使用现行权重，二者互不影响。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/healthz` | 存活探针 |
| GET | `/v1/models` | 可用设备型号及权重大小 |
| POST | `/v1/infer` | 提交测值，返回异常分数（仅现行权重） |
| POST | `/v1/compare` | 同一段测值对照现行权重与指定候选版本，返回双版本分数、判定与分数差 |
| GET | `/v1/cache` | 预算/驻留/预留/在途/计数快照（含候选条目） |
| GET | `/docs` | Swagger UI |

错误统一为 `{"error": {"code": ..., "message": ..., ...}}`：

| 状态码 | code | 场景 |
| --- | --- | --- |
| 404 | `unknown_model` | 型号不在清单中 |
| 404 | `unknown_candidate` | 候选目录中没有该型号的指定候选版本 |
| 422 | `invalid_measurement` | 空数组、非有限值、维度错误 |
| 500 | `model_load_failed` / `inference_failed` | 权重（含候选权重）读取或校验失败 |
| 503 | `model_busy` | 预算不足且所有驻留模型都有在途推理；对照接口在空闲预算放不下候选时同样返回；响应带 `Retry-After` |
| 507 | `model_too_large` | 单模型权重超过整个预算，永远不会成功 |

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_CACHE_BYTES` | `640` | 驻留权重 + 装载预留的上限（字节），现行权重与候选权重共用。每套模型权重 296 字节，640 可容纳两套，第三套会触发 LRU 淘汰 |
| `INFERENCE_DELAY_MS` | `250` | 模拟推理耗时，便于在 Swagger 中观察并发行为 |
| `MODEL_LOAD_DELAY_MS` | `0` | 模拟权重读取耗时，便于复现并发冷装载 |
| `MODEL_DIR` | `./models` | 生产模型清单与权重文件目录（`/v1/infer` 唯一数据源） |
| `CANDIDATE_MODEL_DIR` | `./candidate_models` | 独立候选权重目录，仅 `/v1/compare` 读取；Compose 以只读绑定挂载提供 |
| `INFERENCE_PORT` | `8000` | Compose 暴露的主机端口 |

预算只统计数组形式的权重（`coefficients`、`feature_mean`、`feature_scale`、`intercept`）。
清单里的 `weights_bytes` 是预留依据；装载后实际字节必须与其一致，否则视为装载失败。
候选目录使用独立的 `candidate_manifest.json`（`schema_version: 1`，`candidates` 数组），
每个条目包含 `model_id` / `version` / `file` / `weights_bytes`；候选权重文件与生产权重执行同一套校验
（特征 schema、有限数组、正数 `feature_scale`、阈值范围、字节一致性），文件路径限制为目录内普通文件名。
候选清单在每次对照请求时重新读取，发布新候选版本无需重启服务。

## 候选对照的并发与内存语义

实现集中在 `app/cache.py` 与 `app/candidates.py`，候选条目与正式条目共用同一个 `ModelCache` 与同一字节预算：

1. **总占用不超预算**：现行与候选的 `resident_bytes + reserved_bytes` 之和在每次状态变更时都不超过 `MODEL_CACHE_BYTES`。
2. **对照不驱逐正式模型**：候选装载只允许使用当前空闲预算，不触发任何淘汰；预算放不下候选时直接返回 503
   （`candidate_busy_rejections` 计数），驻留的正式模型与其他候选保持原样。
3. **候选空闲后让位正式推理**：正式冷装载需要空间时，先按 LRU 淘汰空闲候选（`inflight == 0` 且无等待者），
   再淘汰空闲正式模型；在途对照的候选持有租约，不可被淘汰，此时空间仍不足则正式请求得到 503。
4. **双版本在途占用**：一次对照同时持有现行与候选两个租约，两段计算都结束后才分别释放，
   因此断连不会提前释放占用，也不会出现只算完一个版本的"半份结果"。
5. **断连语义与正式推理一致**：等待装载时断连只移除当前等待者，共享的单飞装载任务（含候选装载）继续运行；
   计算开始后断连，请求立即结束但两段计算在线程池中跑完，租约由计算任务自行释放。
6. **候选装载失败**：校验不通过或文件缺失时删除条目并归还预留，返回 500；其他型号与正式推理不受影响，
   修复文件后下一次请求会重新尝试。
7. **单飞与预算规则不变**：同型号同版本的并发对照共用一次候选装载；其他型号继续按原有预算、LRU 与 503 规则服务。

## 并发与内存语义（现行模型）

实现集中在 `app/cache.py`：

1. **总占用不超预算**：`resident_bytes + reserved_bytes <= MODEL_CACHE_BYTES` 在每次状态变更时成立。
2. **单飞装载**：冷请求在缓存锁内登记一条 `LOADING` 记录并创建独立装载任务；后续同模型请求只等待同一个事件，
   实测 10 个并发冷请求只产生 1 次 `load_attempt`。
3. **等待装载时断连**：请求方只取消自己的等待，不触碰装载任务；其他等待者仍能拿到模型。
   装载任务归缓存所有，最后一个等待者离开也不会取消它。
4. **推理中断连**：请求可以立即结束，但计算在线程池中继续，模型租约由计算任务在真正结束后释放，
   期间该模型不可被淘汰。
5. **淘汰策略**：只考虑 `inflight == 0` 且没有等待者的驻留模型，空闲候选优先于空闲正式模型，同层级按 LRU 顺序；
   若这些模型的全部空间仍不够，在淘汰任何模型之前就返回 503，避免"淘汰了却仍然装不下"。
6. **装载失败**：删除记录并归还预留，其他模型不受影响；下次请求会重试该模型。

## 模型

`models/` 下是三份固定权重逻辑回归（`pump` / `fan` / `compressor`）与 `manifest.json`，
由 `scripts/generate_models.py` 确定性生成。特征为振动与电流各自的
均值、RMS、峰值、标准差、波峰因数、峭度，共 12 维；模型各自标准化后做点积并取 sigmoid。
服务只读取权重，不做任何在线训练。

`candidate_models/` 是独立的候选目录，包含 `candidate_manifest.json` 与三份
`1.1.0-rc1` 候选权重（同样 296 字节、同样 12 维特征），由
`scripts/generate_candidate_models.py` 确定性生成，仅用于 `/v1/compare`。
发布新候选版本时，在候选清单中登记新的 `(model_id, version, file, weights_bytes)` 条目并放入权重文件即可，
服务下一次对照请求即会读到，无需重启；Docker Compose 通过只读绑定挂载 `./candidate_models` 提供该目录。

## 本地开发与测试

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt pytest pytest-asyncio httpx pyyaml
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app.api:app --reload
```

测试覆盖：特征抽取边界、生产清单/权重校验、候选清单/权重校验与目录穿越防御、预算与预留、
LRU 淘汰（含候选优先让位）、忙拒绝、单飞装载（生产与候选）、装载失败恢复、对照接口的双版本分数与
`decision_changed`、Swagger 发布，以及在 ASGI 层注入 `http.disconnect` 的等待中断连、计算中断连与
对照/正式推理并发交错语义。
