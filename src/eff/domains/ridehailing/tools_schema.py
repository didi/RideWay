"""Tool schema definitions for the ride-hailing domain."""

from typing import Dict, Any
from eff.utils.schema_utils import create_tool_schema_manager

# ─────────────────────────────────────────────────────────────
# 价格语义统一定义（PRICE_SEMANTICS）
# 所有涉及价格的角色（agent / user-llm / evaluator / rubric 生成）共同遵守此定义
# ─────────────────────────────────────────────────────────────
PRICE_SEMANTICS_ZH = """\
[价格定义 · 所有角色共同遵守]
- estimate_fee = 预估原价（基础费 + 距离 × 单价），**不含优惠 / 动态加价**
- discount_fee = 实际优惠金额（每司机/订单不同，由工具返回）
- surge_multiplier = 动态加价倍率（由 get_ride_real_time_status 返回，默认 1.0）
- **实付价格 = (estimate_fee − discount_fee) × surge_multiplier**
- 涉及预算 / 价格上限判定（如"预算 50 内"），**统一按实付价格比对**，不按预估原价
"""

PRICE_SEMANTICS_EN = """\
[Price Semantics · shared across all roles]
- estimate_fee = base price (base_fee + distance × per_km), **without discount / surge**
- discount_fee = actual discount (per-driver/order, returned by tool)
- surge_multiplier = dynamic surge multiplier (from get_ride_real_time_status, default 1.0)
- **Actual paid = (estimate_fee − discount_fee) × surge_multiplier**
- For budget / price-cap comparisons (e.g. "within 50"), **always compare against actual paid**, not base price
"""

TOOL_DESCRIPTIONS_ZH = {
    "get_home_address": {
        "description": "获取用户的家地址信息，包括地址名称、详细地址、经纬度等",
        "preconditions": "处于打车场景，用户提到要去家或从家出发",
        "postconditions": "返回家地址的详细信息",
        "args": {
            "user_id": "用户ID"
        },
        "returns": "家地址的POI信息（包含地址名称、详细地址、经纬度等）",
        "tool_type": "READ"
    },

    "get_company_address": {
        "description": "获取用户的公司地址信息，包括地址名称、详细地址、经纬度等",
        "preconditions": "处于打车场景，用户提到要去公司或从公司出发",
        "postconditions": "返回公司地址的详细信息",
        "args": {
            "user_id": "用户ID"
        },
        "returns": "公司地址的POI信息（包含地址名称、详细地址、经纬度等）",
        "tool_type": "READ"
    },

    "select_car_types": {
        "description": "让用户选择打车的车型。合法车型仅限 [特惠/快车/特快/出租车/优享/专车/六座专车/豪华车/六座豪华车]（按实付价格升序）",
        "preconditions": "处于打车场景，用户需要选择或切换车型",
        "postconditions": "返回已选择的车型列表",
        "args": {
            "car_types": "车型列表，合法值仅限 [特惠/快车/特快/出租车/优享/专车/六座专车/豪华车/六座豪华车]"
        },
        "returns": "选择车型的结果",
        "tool_type": "WRITE"
    },

    "get_available_car_types": {
        "description": "获取当前可用的车型列表",
        "preconditions": "处于打车场景，需要了解有哪些车型可选",
        "postconditions": "返回可用车型列表",
        "args": {},
        "returns": "可用车型列表",
        "tool_type": "READ"
    },

    "match_drivers": {
        "description": "根据起终点和车型/品牌匹配可用司机，返回司机列表（含司机ID、姓名、车牌、车型、品牌车系、司机评分、标签、OTA、OEA、estimate_fee、discount_fee 等）。当用户提出品牌偏好（如『要奔驰/宝马/奥迪/特斯拉等』），通过 car_brands 参数传入；工具会优先按品牌过滤，若无完全匹配会按近似返回并标注。【价格语义】estimate_fee 是原价、discount_fee 是该司机优惠，实付 = estimate_fee − discount_fee（如有 surge 再乘倍率）；预算判定按实付，不按原价。",
        "preconditions": "已确认起终点地址，需要匹配司机",
        "postconditions": "返回司机列表",
        "args": {
            "user_id": "用户ID",
            "start_address": "起点地址",
            "end_address": "终点地址",
            "car_types": "车型筛选列表（可选）。合法值仅限 [特惠/快车/特快/出租车/优享/专车/六座专车/豪华车/六座豪华车]（按实付价格升序），传非法值会报错",
            "car_brands": "品牌偏好列表（可选），如 ['奔驰','宝马','奥迪','特斯拉','比亚迪','理想','大众']。工具按品牌字段做包含匹配，过滤后若 ≥3 个则按筛选结果返回，<3 个会补充近似司机"
        },
        "returns": "匹配到的司机列表，第一行会注明是否按品牌偏好筛选；每个司机含 driver_id、driver_name、plate_no、car_type、brand_desc、car_model、recommend_score、tags、estimate_fee、discount_fee 等",
        "tool_type": "READ"
    },

    "create_ride_order": {
        "description": "创建打车订单，包含订单ID、司机ID、车型、司机评分、标签、OTA、OEA、estimate_fee、discount_fee 等；支持途经点。【价格语义】estimate_fee 原价、discount_fee 优惠；实付 = estimate_fee − discount_fee（如有 surge 再乘倍率）；预算判定按实付。可多次调用以创建多笔相互独立的订单。",
        "preconditions": "处于打车场景，用户已选择司机，确认起终点",
        "postconditions": "返回创建的打车订单信息，等待用户确认用车",
        "args": {
            "user_id": "用户ID",
            "driver_id": "司机ID",
            "start_address": "起点地址",
            "end_address": "终点地址",
            "via_addresses": "途经点地址列表（可选），按经过顺序传入；用户提到途经点时必须传入"
        },
        "returns": "创建成功返回订单详情（含order_id、driver_id、car_type、recommend_score、tags、ota、oea、estimate_fee、discount_fee、via_pois等），否则返回错误信息",
        "tool_type": "WRITE"
    },

    "confirm_ride": {
        "description": "确认用车，将订单状态从待确认变为已确认",
        "preconditions": "处于打车场景，订单已创建，用户表达确认用车",
        "postconditions": "返回确认结果，司机开始前来接驾",
        "args": {
            "order_id": "订单ID"
        },
        "returns": "确认用车的结果信息",
        "tool_type": "WRITE"
    },

    "cancel_ride": {
        "description": "取消用车，禁止对已取消的订单再次取消",
        "preconditions": "处于打车场景，用户表达取消用车",
        "postconditions": "返回取消结果信息",
        "args": {
            "order_id": "订单ID",
            "reason": "取消原因（可选）"
        },
        "returns": "取消用车的结果信息",
        "tool_type": "WRITE"
    },

    "add_ride_note": {
        "description": "给打车订单添加备注信息，如特殊要求（需要等候、携带大件物品、宠物、排除车型等）、乘客说明、司机提醒等",
        "preconditions": "订单已创建",
        "postconditions": "备注信息已添加到订单，司机可见",
        "args": {
            "order_id": "订单ID",
            "note": "备注内容（如'携带婴儿车需要大后备箱'、'上车等5分钟装行李'、'不要电车'等）"
        },
        "returns": "备注添加结果，包含当前所有备注",
        "tool_type": "WRITE"
    },

    "get_ride_order_detail": {
        "description": "获取打车订单详情，返回订单的完整信息（含 estimate_fee 原价 + discount_fee 优惠）。【价格语义】实付 = estimate_fee − discount_fee（如有 surge 再乘倍率），预算判定按实付。",
        "preconditions": "上文确定唯一一个打车order_id",
        "postconditions": "返回指定订单的详细信息",
        "args": {
            "order_id": "订单ID"
        },
        "returns": "订单详细信息，包含order_id、driver_id、car_type、estimate_fee、discount_fee、status等",
        "tool_type": "READ"
    },

    "get_ride_estimate": {
        "description": "获取打车预估价格（原价）和最高优惠，按车型分类展示；支持途经点（多目的地按累计距离计价）。返回中 \"预估价格\" 是原价，实付 = 预估价格 - 实际优惠（≤ 最高优惠）。当用户对比『途经 A 还是 B 顺路且便宜』时，分别传不同 via_addresses 调用本工具比较各方案在不同车型下的价格；价格仅与累计距离/车型相关，与 match_drivers 当前可用司机的优惠无关。",
        "preconditions": "处于打车场景，已确定起终点地址；多目的地比价时确定各候选途经方案",
        "postconditions": "返回该路径（含途经累计）下各车型的预估原价和最高优惠信息",
        "args": {
            "start_address": "起点地址",
            "end_address": "终点地址",
            "via_addresses": "可选途经点列表（list[str]），按经过顺序传入；行程距离 = 起 → 途经1 → ... → 终 累计 km，价格随之变化。评估『哪条路线最便宜』时分别传不同 via_addresses 对比"
        },
        "returns": "返回路径字符串、累计距离和各车型预估价格/优惠信息",
        "tool_type": "READ"
    },

    "search_ride_orders": {
        "description": "查询用户的打车订单列表",
        "preconditions": "按照查询条件查看用户打车订单",
        "postconditions": "返回符合条件的打车订单列表",
        "args": {
            "user_id": "用户ID",
            "status": "订单状态筛选（可选），如 waiting_confirm、confirmed、cancelled、scheduled"
        },
        "returns": "返回符合条件的打车订单详细信息",
        "tool_type": "READ"
    },

    "create_scheduled_ride": {
        "description": "创建预约打车订单，按用户指定的未来时间出发（出发时间距当前超过5分钟）；支持途经点。可多次调用以创建多笔相互独立的预约单（不同出发时间各一笔，可同时存在）。",
        "preconditions": "处于打车场景，用户明确指定了超过5分钟后的出发时间",
        "postconditions": "返回预约订单详情，状态为 scheduled",
        "args": {
            "user_id": "用户ID",
            "driver_id": "司机ID",
            "start_address": "起点地址",
            "end_address": "终点地址",
            "departure_time": "预约出发时间，参数名严格用 'departure_time'（不是 scheduled_time / start_time / time）。格式 YYYY-MM-DD HH:MM 或 YYYY-MM-DD HH:MM:SS。",
            "via_addresses": "途经点地址列表（可选），按经过顺序传入；用户提到途经点时必须传入"
        },
        "returns": "预约订单详情（含 order_id、departure_time、status=scheduled、via_pois 等）",
        "tool_type": "WRITE"
    },

    "create_ride_reminder": {
        "description": "创建一个行程提醒：在指定时间提醒用户某段行程需要叫车或预约。仅记录提醒信息（提醒时间、起终点、途经点），不会自动下单，也不会匹配司机。可多次调用以登记多条相互独立的提醒（不同提醒时间 / 不同行程各一条，可同时有效）。",
        "preconditions": "用户某段行程暂不立即下单，希望稍后在某个时间点再被提醒去叫车/预约",
        "postconditions": "返回提醒创建结果（含 reminder_id 与提醒内容）",
        "args": {
            "user_id": "用户ID",
            "remind_time": "触发提醒的时间，格式 YYYY-MM-DD HH:MM 或 YYYY-MM-DD HH:MM:SS",
            "start_address": "该段行程的起点地址",
            "end_address": "该段行程的终点地址",
            "via_addresses": "途经点地址列表（可选），按经过顺序传入",
            "note": "提醒备注（可选），如该段行程的特别说明"
        },
        "returns": "提醒创建结果（含 reminder_id、remind_time、起终点、途经点、备注）",
        "tool_type": "WRITE"
    },

    "get_ride_real_time_status": {
        "description": "获取已确认订单的实时动态，包括司机当前位置、预计到达时间、是否发生异常事件（司机取消、溢价变化、ETA大幅延长）",
        "preconditions": "订单已确认（confirmed 或 scheduled 状态），需要主动感知实时变化",
        "postconditions": "返回实时状态信息，event 字段指示异常类型",
        "args": {
            "order_id": "订单ID"
        },
        "returns": "实时状态：driver_eta（秒）、driver_distance（米）、surge_multiplier（溢价倍数，1.0=无溢价）、event（none/driver_cancelled/eta_delay/surge_change）、event_detail（事件说明）",
        "tool_type": "READ"
    },

    "update_ride_destination": {
        "description": "修改打车订单的目的地（终点），适用于用户在确认前或行程中变更目的地",
        "preconditions": "订单存在且状态为 waiting_confirm、confirmed 或 scheduled；用户请求变更终点",
        "postconditions": "返回更新后的订单信息，终点已变更",
        "args": {
            "order_id": "订单ID",
            "new_end_address": "新目的地地址"
        },
        "returns": "更新后的订单详情（含新的 end_poi 信息）",
        "tool_type": "WRITE"
    },

    "get_history_orders": {
        "description": "获取用户历史打车订单（已完成的历史记录），可用于了解用户常用车型偏好，在用户未明确指定车型时作为推荐依据。典型触发场景：用户用指代性表述（「老地方」/「上次那家」/「常去的」/「按习惯」/「按平时」）时，应调用本工具推断目的地或车型偏好。【价格语义】返回的 estimate_fee/discount_fee 同 match_drivers，实付按 estimate_fee − discount_fee 计算。",
        "preconditions": "需要了解用户历史打车习惯；或用户未指定车型时推荐合适车型；或用户用指代性表述（老地方/上次/常去）需要推断具体目的地",
        "postconditions": "返回历史订单列表，包含车型、起终点、费用等信息",
        "args": {
            "user_id": "用户ID"
        },
        "returns": "历史打车订单列表（含车型、起终点、费用等）",
        "tool_type": "READ"
    },

    "get_current_location": {
        "description": "获取用户当前位置的地址和经纬度，适用于用户直接说「去某地」而未提供起点时，自动以当前位置作为起点",
        "preconditions": "用户未明确指定起点地址，需要以当前位置作为打车起点",
        "postconditions": "返回当前位置的地址、经纬度信息",
        "args": {},
        "returns": "当前位置信息（address、longitude、latitude）",
        "tool_type": "READ"
    },

    "get_flight_info": {
        "description": "根据航班号查询航班起降信息（起飞城市/机场/时间 + 到达城市/机场/时间）。【何时用】用户提到具体航班号，需要：①安排接机（起点=到达机场，时间=航班到达时间）②安排送机（终点=起飞机场，反推出发时间）③推断航班到达后回家叫车的起终点和时间。【何时不用】user 未给出具体航班号时，**不要**调用本工具（不要凭空猜测航班号），应直接走时间反推路径。【返回】航班号 + 起飞(time, city, airport) + 到达(time, city, airport)。",
        "preconditions": "用户给出明确航班号，且需要根据航班信息决策叫车的起终点 / 时间",
        "postconditions": "返回该航班的起降城市、机场航站楼、起飞与到达时间",
        "args": {
            "flight_no": "用户对话中实际给出的航班号；user 未提供时不要调用本工具"
        },
        "returns": "航班起降信息文本：起飞 time/city/airport + 到达 time/city/airport",
        "tool_type": "READ"
    },

    "nearby_search": {
        "description": "在指定参考点附近搜索 POI 候选，返回名称、经纬度和距离（km）。【何时用】用户提到品类词（如『最近的咖啡店』『家附近的药店』『附近三甲医院』）需要找到具体 POI 时。【何时不用】用户已给具体地名 → 用 address_to_longitude_latitude 或 search_pois；若用户表达『顺路 / 路上 / 沿途』→ 用 search_along_route。【返回】按距离升序的 top 5 候选；agent 可基于此选最近或反问用户挑选。**返回的候选会自动持久化到 db.location**，后续 via_addresses 直接传 POI 名即可，无需再调 address_to_longitude_latitude。",
        "preconditions": "用户表达需要搜索某类附近 POI，且能确定参考点（当前位置/某地名/某坐标）",
        "postconditions": "返回 POI 候选列表（含距离）；同时将 top 候选写入 db.location 供后续工具引用",
        "args": {
            "query": "要搜索的内容（品类词），如 '咖啡馆'、'药店'、'三甲医院'、'火锅店'、'电影院' 等",
            "location": "可选 dict 指定参考点，三选一：{\"city_name\": \"海口\"} / {\"place_name\": \"腾讯滨海大厦\"} / {\"lng\": 110.32, \"lat\": 20.04}；未指定时以 task 当前位置为参考"
        },
        "returns": "候选 POI 文本列表（name/lng/lat/distance_km），按距离升序 top 5",
        "tool_type": "READ"
    },

    "search_along_route": {
        "description": "沿途搜：在起点→终点路线上搜索 POI 候选，按绕路距离升序返回。【何时用】用户表达『顺路』『路上』『沿途』『下班顺路买X』『去公司路上拿X』等场景——POI 应在行进方向上的合理范围内，而不是某个端点附近。【何时不用】若 POI 在某一端附近且不强调顺路（『家附近的药店』『公司楼下的咖啡』）→ 用 nearby_search 更合适。【返回】绕路距离 ≤ max_detour_km 的 top 5 候选，按绕路升序；自动持久化到 db.location，后续 via_addresses 可直接引用。【与 nearby_search 的区别】nearby_search 度量 POI 到单点的距离；search_along_route 度量 POI 是否在行程线上（绕路 = (起→POI→终) − (起→终直线)）。",
        "preconditions": "用户表达顺路/沿途某类 POI 的需求，且起终点都已明确（已通过 search_pois / nearby_search / address_to_longitude_latitude 等可解析）",
        "postconditions": "返回沿路 POI 候选列表（含绕路距离 km）；同时将 top 候选写入 db.location 供后续工具引用",
        "args": {
            "query": "搜索内容（品类词 / 品牌名 / 具体店名），如 '咖啡店'、'加油站'、'鲜花店'、'星巴克' 等",
            "start_address": "起点地址（须可被 _lookup_coords 解析）",
            "end_address": "终点地址（须可被 _lookup_coords 解析）",
            "max_detour_km": "可选，允许的最大绕路距离（km），默认 2.0；放宽以容纳更远候选"
        },
        "returns": "沿路 POI 候选列表（name/lng/lat/绕路 km/全程 km），按绕路升序 top 5",
        "tool_type": "READ"
    },

    "address_to_longitude_latitude": {
        "description": "根据**已明确单一**的地址名解析经纬度，返回单个最佳匹配 [lng, lat]。优先查 task 环境 POI（start_poi / end_poi / location / home / company），未命中回退全局地址库。【何时用】用户已给精确具体地址，你只需拿坐标做距离/路程计算时。【何时不用】地名含歧义 / 品类词 / 连锁多店 / 多航站楼/入口 → 改用 search_pois 列候选反问。【决策树】明确单点→本工具；歧义/品类/多店→search_pois；不确定→先 search_pois 看候选数（=1 可继续，>1 必反问）。",
        "preconditions": "用户给出的地址明确单一无歧义，仅需获取坐标",
        "postconditions": "返回 [lng, lat] 单一坐标对（未命中时回退全局地址库）",
        "args": {
            "address": "明确单一的地址名"
        },
        "returns": "[lng, lat] 坐标对",
        "tool_type": "READ"
    },

    "search_pois": {
        "description": "搜索匹配查询关键词的所有 POI 候选，返回多个候选项以支持地址歧义消解。当用户说一个可能有多重含义的地点（如「万达广场」可能指多个城市的店）时，使用本工具能看到所有候选，主动向用户澄清。【何时用】用户给的地名含歧义 / 品类无修饰词 / 多门店连锁无门店名 / 多航站楼或多入口等。【何时不用】用户已给明确单一精确地址 → 改用 address_to_longitude_latitude（只返回单坐标，无需 agent 反问）。【决策树】明确单点→address_to_longitude_latitude；歧义/品类/多店→本工具；本工具返回 >1 个候选时**必须**向用户反问确认，返回 =1 个时可直接采用（参见 C5 维度）。典型触发场景：用户只说「机场/火车站/高铁站/咖啡店/医院/花店/便利店/加油站」等品类无修饰词、或多门店连锁（「麦当劳/星巴克」无门店名）、或多航站楼/多入口（「去机场」未指定 T1/T2）。",
        "preconditions": "用户给出的地点名称可能有歧义（多个 POI 匹配），需要列出候选让用户选择；包括品类词无具体修饰、多门店连锁无门店名、多航站楼/多入口等场景",
        "postconditions": "返回 POI 候选列表（name、address、lng、lat、source、相似度），多个候选时会提示存在歧义",
        "args": {
            "query": "要搜索的地点名称",
            "limit": "返回候选数量上限，默认 5"
        },
        "returns": "POI 候选列表（含名称、坐标、来源、相似度），多于 1 个时建议向用户澄清",
        "tool_type": "READ"
    },

    "get_trip_duration_estimate": {
        "description": "估算从起点 → [途经点] → 终点的路程时间（分钟）和距离（公里）。用于反向时间推理（如明早 7 点航班要几点叫车）和多目的地路线评估（如比较『A 起点 → 途经 B → 终点』vs『B 起点 → 途经 A → 终点』哪条更顺路）。【路况说明】高峰判定自动基于 system_time（早 7-9 / 晚 17-19 点 → 1.5x 拥堵），agent **无需也无法**通过参数控制高峰；同 task 任何调用同输入必同输出。区别于 match_drivers（只给当前可用司机接驾 ETA，不适合未来时段或途经点对比）。典型触发场景：用户提到「X 点前到达 Y 地」「赶 Z 点的航班/高铁/会议」等到达时间约束时，agent 应调本工具拿到耗时后自行反推出发时间。",
        "preconditions": "需要预测某段路程（含可选途经点）的耗时；多候选起点/途经点选择时用本工具对比；用户提到到达时间约束（赶飞机/高铁/会议）需要反推出发时间",
        "postconditions": "返回总距离 km（含途经点累加）、预计耗时 min、是否高峰、推荐缓冲",
        "args": {
            "start_address": "起点地址",
            "end_address": "终点地址",
            "via_addresses": "可选途经点列表（list[str]），按顺序计算 起 → 途经1 → 途经2 → ... → 终 的总距离；评估『哪个候选点做途经』时分别传不同 via_addresses 比较"
        },
        "returns": "路程预估文本（含完整路径、总距离、耗时、高峰判断、推荐缓冲）",
        "tool_type": "READ"
    },
}

TOOL_DESCRIPTIONS_EN = {
    "get_home_address": {
        "description": "Get user's home address including name, detailed address, and coordinates",
        "preconditions": "In ride-hailing scenario, user mentions going home or departing from home",
        "postconditions": "Return detailed home address information",
        "args": {
            "user_id": "User ID"
        },
        "returns": "Home address POI information (name, address, coordinates)",
        "tool_type": "READ"
    },

    "get_company_address": {
        "description": "Get user's company address including name, detailed address, and coordinates",
        "preconditions": "In ride-hailing scenario, user mentions going to or departing from company",
        "postconditions": "Return detailed company address information",
        "args": {
            "user_id": "User ID"
        },
        "returns": "Company address POI information (name, address, coordinates)",
        "tool_type": "READ"
    },

    "select_car_types": {
        "description": "Allow user to select car types. Legal values only: [特惠/快车/特快/出租车/优享/专车/六座专车/豪华车/六座豪华车] (ordered by actual paid, ascending)",
        "preconditions": "In ride-hailing scenario, user needs to select or switch car types",
        "postconditions": "Return selected car types list",
        "args": {
            "car_types": "Car type list, legal values only: [特惠/快车/特快/出租车/优享/专车/六座专车/豪华车/六座豪华车]"
        },
        "returns": "Car type selection result",
        "tool_type": "WRITE"
    },

    "get_available_car_types": {
        "description": "Get list of available car types",
        "preconditions": "In ride-hailing scenario, need to know available car types",
        "postconditions": "Return available car types list",
        "args": {},
        "returns": "Available car types list",
        "tool_type": "READ"
    },

    "match_drivers": {
        "description": "Match available drivers based on start/end points and car types, return driver list with driver_id, name, plate_no, car_type, recommend_score, tags, ota, oea, estimate_fee, discount_fee",
        "preconditions": "In ride-hailing scenario, start and end addresses confirmed, need to match drivers",
        "postconditions": "Return matched driver list, guide user to select a driver",
        "args": {
            "user_id": "User ID",
            "start_address": "Start address",
            "end_address": "End address",
            "car_types": "Car type filter list (optional). Legal values only: [特惠/快车/特快/出租车/优享/专车/六座专车/豪华车/六座豪华车] (ordered by actual paid, ascending); illegal values are rejected"
        },
        "returns": "Matched driver list with driver_id, driver_name, plate_no, car_type, recommend_score, tags, ota, oea, estimate_fee, discount_fee",
        "tool_type": "READ"
    },

    "create_ride_order": {
        "description": "Create ride order with order_id, driver_id, car_type, recommend_score, tags, ota, oea, estimate_fee, discount_fee; supports waypoints (via points). Can be called multiple times to create several independent orders.",
        "preconditions": "In ride-hailing scenario, user has selected a driver and confirmed start/end points",
        "postconditions": "Return created ride order info, wait for user to confirm ride",
        "args": {
            "user_id": "User ID",
            "driver_id": "Driver ID",
            "start_address": "Start address",
            "end_address": "End address",
            "via_addresses": "Optional list of waypoint addresses in visiting order; must be provided when the user mentions waypoints"
        },
        "returns": "Order details including order_id, driver_id, car_type, recommend_score, tags, ota, oea, estimate_fee, discount_fee, via_pois",
        "tool_type": "WRITE"
    },

    "confirm_ride": {
        "description": "Confirm ride, change order status from waiting_confirm to confirmed",
        "preconditions": "In ride-hailing scenario, order created, user confirms ride",
        "postconditions": "Return confirmation result, driver starts coming to pick up",
        "args": {
            "order_id": "Order ID"
        },
        "returns": "Ride confirmation result",
        "tool_type": "WRITE"
    },

    "cancel_ride": {
        "description": "Cancel ride. Prohibited from canceling orders already in cancelled status",
        "preconditions": "In ride-hailing scenario, user expresses cancellation",
        "postconditions": "Return cancellation result",
        "args": {
            "order_id": "Order ID",
            "reason": "Cancellation reason (optional)"
        },
        "returns": "Ride cancellation result",
        "tool_type": "WRITE"
    },

    "add_ride_note": {
        "description": "Add a note to a ride order for special requirements (e.g. luggage, pets, waiting time, car type exclusions)",
        "preconditions": "Order already created",
        "postconditions": "Note added to order, visible to driver",
        "args": {
            "order_id": "Order ID",
            "note": "Note content (e.g. 'carrying stroller, need large trunk', 'wait 5 min for loading')"
        },
        "returns": "Note addition result with all current notes",
        "tool_type": "WRITE"
    },

    "get_ride_order_detail": {
        "description": "Get ride order details with complete information",
        "preconditions": "Specific ride order_id determined",
        "postconditions": "Return specified order detailed information",
        "args": {
            "order_id": "Order ID"
        },
        "returns": "Order details including order_id, driver_id, car_type, estimate_fee, discount_fee, status",
        "tool_type": "READ"
    },

    "get_ride_estimate": {
        "description": "Get ride fare estimates (base price) and max discount per car type; supports waypoints (multi-stop priced by cumulative distance). When user compares 'which waypoint is more efficient AND cheaper', call this tool with different via_addresses to compare prices by car type. Price depends only on cumulative distance + car type, independent of match_drivers' per-driver discounts.",
        "preconditions": "In ride-hailing scenario, start and end addresses determined; for multi-stop pricing comparison, candidate waypoint plans defined",
        "postconditions": "Return path string, cumulative distance, and fare estimates / max discount per car type",
        "args": {
            "start_address": "Start address",
            "end_address": "End address",
            "via_addresses": "Optional list of waypoint addresses (list[str]) in visiting order; trip distance = start → via1 → ... → end cumulative km, price scales accordingly. Pass different via_addresses lists to compare which route is cheapest."
        },
        "returns": "Path string, cumulative distance, and per-car-type fare/discount info",
        "tool_type": "READ"
    },

    "search_ride_orders": {
        "description": "Search user's ride orders",
        "preconditions": "View ride orders according to query conditions",
        "postconditions": "Return ride orders matching conditions",
        "args": {
            "user_id": "User ID",
            "status": "Order status filter (optional): waiting_confirm, confirmed, cancelled, scheduled"
        },
        "returns": "Detailed information of ride orders matching conditions",
        "tool_type": "READ"
    },

    "create_scheduled_ride": {
        "description": "Create a scheduled ride order with a specific future departure time (more than 5 minutes ahead); supports waypoints. Can be called multiple times to create several independent scheduled orders (one per different departure time; they coexist).",
        "preconditions": "In ride-hailing scenario, user specifies a departure time more than 5 minutes in the future",
        "postconditions": "Return scheduled ride order details with status=scheduled",
        "args": {
            "user_id": "User ID",
            "driver_id": "Driver ID",
            "start_address": "Start address",
            "end_address": "End address",
            "departure_time": "Scheduled departure time. Parameter name MUST be 'departure_time' (NOT scheduled_time / start_time / time). Format: YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM:SS.",
            "via_addresses": "Optional list of waypoint addresses in visiting order; must be provided when the user mentions waypoints"
        },
        "returns": "Scheduled order details including order_id, departure_time, status=scheduled, via_pois",
        "tool_type": "WRITE"
    },

    "create_ride_reminder": {
        "description": "Create a trip reminder: remind the user at a specified time that a certain trip leg needs to be hailed or scheduled. Only records the reminder (remind time, start/end, waypoints); it does NOT place an order or match drivers. Can be called multiple times to register several independent reminders (one per different remind time / trip; they coexist).",
        "preconditions": "A trip leg is not booked immediately; the user wants to be reminded later at a certain time to hail/schedule it",
        "postconditions": "Return reminder creation result (with reminder_id and reminder content)",
        "args": {
            "user_id": "User ID",
            "remind_time": "Time to trigger the reminder, format YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM:SS",
            "start_address": "Start address of this trip leg",
            "end_address": "End address of this trip leg",
            "via_addresses": "Optional list of waypoint addresses in visiting order",
            "note": "Optional reminder note, e.g. special instructions for this leg"
        },
        "returns": "Reminder creation result (with reminder_id, remind_time, start/end, waypoints, note)",
        "tool_type": "WRITE"
    },

    "get_ride_real_time_status": {
        "description": "Get real-time status of a confirmed order including driver location, ETA, and exception events (driver cancelled, surge change, ETA delay)",
        "preconditions": "Order is confirmed or scheduled, need to proactively monitor real-time changes",
        "postconditions": "Return real-time status with event field indicating exception type",
        "args": {
            "order_id": "Order ID"
        },
        "returns": "Real-time status: driver_eta (seconds), driver_distance (meters), surge_multiplier (1.0=no surge), event (none/driver_cancelled/eta_delay/surge_change), event_detail",
        "tool_type": "READ"
    },

    "update_ride_destination": {
        "description": "Update the destination of an existing ride order, for users who change their destination before or during the ride",
        "preconditions": "Order exists with status waiting_confirm, confirmed, or scheduled; user requests destination change",
        "postconditions": "Return updated order details with new end_poi",
        "args": {
            "order_id": "Order ID",
            "new_end_address": "New destination address"
        },
        "returns": "Updated order details with new end_poi",
        "tool_type": "WRITE"
    },

    "get_history_orders": {
        "description": "Get user's historical ride orders (completed history), useful for understanding preferred car types and recommending car types when user has no explicit preference. Typical trigger: when user uses referential expressions like \"old place / last time / usual / as usual\", call this tool to infer destination or car-type preference.",
        "preconditions": "Need to understand user's ride history; or recommend car type when user has no explicit preference; or user uses referential expressions (old place / last time / usual) requiring destination inference",
        "postconditions": "Return list of historical orders including car type, start/end points, and fees",
        "args": {
            "user_id": "User ID"
        },
        "returns": "List of historical ride orders (with car type, start/end points, fees, etc.)",
        "tool_type": "READ"
    },

    "get_current_location": {
        "description": "Get user's current location address and coordinates. Use when user says 'go to X' without specifying a starting point, to use current location as the pickup point",
        "preconditions": "User has not specified a start address; current location is needed as the pickup point",
        "postconditions": "Return current location address and coordinates",
        "args": {},
        "returns": "Current location info (address, longitude, latitude)",
        "tool_type": "READ"
    },

    "get_flight_info": {
        "description": "Query flight departure/arrival info by flight number (departure city/airport/time + arrival city/airport/time). [When to use] User mentions a specific flight number and you need to: ① arrange pickup at arrival airport (start = arrival airport, time = flight arrival time); ② arrange drop-off at departure airport (end = departure airport, back-calculate departure time); ③ infer ride start/end and time for post-flight 'go home' scenarios. [When NOT to use] If user has NOT provided a specific flight number, **do not** call this tool (do not fabricate a flight number); fall back to time back-calculation path instead.",
        "preconditions": "User gives an explicit flight number and ride start/end/time depend on flight info",
        "postconditions": "Return flight's departure/arrival city, airport terminal, and times",
        "args": {
            "flight_no": "Flight number actually provided by the user in dialogue; do not call this tool if user has not provided one"
        },
        "returns": "Flight info text: departure time/city/airport + arrival time/city/airport",
        "tool_type": "READ"
    },

    "nearby_search": {
        "description": "Search nearby POI candidates around a reference point by category keyword, returning name/coordinates/distance (km). [When to use] User mentions a category word (e.g. 'nearest coffee shop', 'pharmacy near home', 'nearby tertiary hospital') and needs to locate a concrete POI. [When NOT to use] User has provided a specific place name → use address_to_longitude_latitude or search_pois; if user expresses 'on the way / along the route' → use search_along_route. [Returns] Top 5 candidates sorted by distance ascending; agent may pick the nearest or ask user to choose. **Top candidates are auto-persisted to db.location** so downstream via_addresses can reference them by name directly.",
        "preconditions": "User expresses need to search a nearby POI category, and a reference point is determinable (current location / a place name / coordinates)",
        "postconditions": "Return POI candidate list with distances; top candidates written to db.location for downstream reference",
        "args": {
            "query": "Category word to search, e.g. 'coffee shop', 'pharmacy', 'tertiary hospital', 'hot pot', 'cinema'",
            "location": "Optional dict specifying reference point, one of: {\"city_name\": \"...\"} / {\"place_name\": \"...\"} / {\"lng\": ..., \"lat\": ...}; defaults to task's current location"
        },
        "returns": "POI candidate list text (name/lng/lat/distance_km), top 5 sorted by distance",
        "tool_type": "READ"
    },

    "search_along_route": {
        "description": "Search POI candidates along a route from start to end, sorted by detour distance ascending. [When to use] User expresses 'on the way / along the route / pick up X on the way home/to office' — POI should lie on the travel path, not just near an endpoint. [When NOT to use] If the POI is near a single endpoint without 'on the way' intent ('pharmacy near home', 'coffee shop next to office') → use nearby_search. [Returns] Top 5 candidates with detour distance ≤ max_detour_km, sorted by detour ascending; auto-persisted to db.location for downstream via_addresses reference. [vs nearby_search] nearby_search measures POI distance to a single point; search_along_route measures whether POI lies on the trip line (detour = (start→POI→end) − (start→end direct)).",
        "preconditions": "User expresses an on-the-way need for some POI category, and both endpoints are resolvable",
        "postconditions": "Return route-POI candidates with detour distance; top candidates written to db.location for downstream reference",
        "args": {
            "query": "Search content (category / brand / specific store name), e.g. 'coffee shop', 'gas station', 'florist', 'Starbucks'",
            "start_address": "Start address (must be resolvable by _lookup_coords)",
            "end_address": "End address (must be resolvable by _lookup_coords)",
            "max_detour_km": "Optional, max allowed detour distance in km, default 2.0; loosen to include further candidates"
        },
        "returns": "Route-POI candidates (name/lng/lat/detour km/total km), top 5 sorted by detour ascending",
        "tool_type": "READ"
    },

    "address_to_longitude_latitude": {
        "description": "Resolve a single explicit address to coordinates, returning the single best-match [lng, lat]. Searches task POIs first (start_poi / end_poi / location / home / company), falling back to global address book. [When to use] User has given a precise specific address and you only need coordinates for distance/route calc. [When NOT to use] Name contains ambiguity / category word / chain multi-store / multi-terminal-or-entrance → use search_pois instead to list candidates and ask clarification. [Decision tree] Explicit single point → this tool; ambiguous/category/multi-store → search_pois; uncertain → search_pois first to see candidate count (=1 can proceed, >1 must ask).",
        "preconditions": "User-given address is explicit, single, and unambiguous; only coordinates needed",
        "postconditions": "Return [lng, lat] single coordinate pair (falls back to global address book if not found)",
        "args": {
            "address": "Explicit single address name"
        },
        "returns": "[lng, lat] coordinate pair",
        "tool_type": "READ"
    },

    "search_pois": {
        "description": "Search all POI candidates matching the query keyword. Returns multiple candidates to enable address ambiguity resolution. When the user mentions an ambiguous location (e.g. 'Wanda Plaza' could refer to multiple stores), use this to surface all candidates and proactively clarify with the user. [When to use] Name is ambiguous / category word w/o modifier / multi-store chain w/o store name / multi-terminal-or-entrance. [When NOT to use] User has given an explicit single precise address → use address_to_longitude_latitude (single coord, no clarification needed). [Decision tree] Explicit single point → address_to_longitude_latitude; ambiguous/category/multi-store → this tool; when this tool returns >1 candidate, you **must** ask the user to clarify; when =1 candidate, you may adopt directly (see C5 dimension). Typical triggers: user says only a category word without modifier (\"airport / train station / coffee shop / hospital / florist / convenience store / gas station\"), or chain brand without specific store name (\"McDonald's / Starbucks\"), or multi-terminal/multi-entrance location (\"to airport\" without T1/T2).",
        "preconditions": "User-provided location may be ambiguous (multiple POIs could match); covers category-word-without-modifier, chain-without-store-name, multi-terminal/multi-entrance scenarios",
        "postconditions": "Return POI candidate list (name, address, lng, lat, source, similarity); when more than 1 match, prompt for clarification",
        "args": {
            "query": "Location name to search for",
            "limit": "Max number of candidates, default 5"
        },
        "returns": "POI candidate list (with name, coords, source, similarity); suggests clarification if >1 match",
        "tool_type": "READ"
    },

    "get_trip_duration_estimate": {
        "description": "Estimate trip duration (minutes) and distance (km) from start through optional waypoints to end. Used for backward time reasoning (e.g. 'flight at 7am, when should I call a ride?') and multi-stop route evaluation. [Traffic note] Peak-hour judgement is automatic based on system_time (morning 7-9 / evening 17-19 → 1.5x congestion); agent CANNOT control peak via parameters — same task, same input always gives same output. Different from match_drivers (which only gives current available drivers' pickup ETA, not suitable for future trip duration or waypoint comparison). Typical trigger: when user mentions an arrival deadline (\"arrive at Y by X\" / \"catch the flight at Z\" / \"meeting at noon\"), agent calls this tool to get duration then back-calculates departure time itself.",
        "preconditions": "Need to predict trip duration (especially for scheduled-ride backward time reasoning); also used to compare candidate routes via different waypoints; or user mentions arrival deadline (catching flight/train/meeting) requiring departure-time back-calculation",
        "postconditions": "Return total distance km (including waypoints), estimated duration min, peak-hour flag, recommended buffer time",
        "args": {
            "start_address": "Start address",
            "end_address": "End address",
            "via_addresses": "Optional list of waypoint addresses (list[str]) in visiting order; total distance = start → via1 → via2 → ... → end. Pass different via_addresses lists to compare which candidate makes the better waypoint."
        },
        "returns": "Trip duration estimate text (includes full path, total distance, duration, peak-hour judgement, recommended buffer)",
        "tool_type": "READ"
    },
}

_schema_manager = create_tool_schema_manager("ridehailing", TOOL_DESCRIPTIONS_ZH, TOOL_DESCRIPTIONS_EN)


def get_tool_descriptions():
    return _schema_manager.get_tool_descriptions()


def get_tool_description(tool_name: str) -> Dict[str, Any]:
    return _schema_manager.get_tool_description(tool_name)


def get_all_tool_names() -> list:
    return _schema_manager.get_all_tool_names()


def get_tools_by_type(tool_type: str) -> Dict[str, Dict[str, Any]]:
    return _schema_manager.get_tools_by_type(tool_type)


def get_tool_count_by_type() -> Dict[str, int]:
    return _schema_manager.get_tool_count_by_type()


def get_tool_args(tool_name: str) -> Dict[str, str]:
    return _schema_manager.get_tool_args(tool_name)


def get_tool_returns(tool_name: str) -> str:
    return _schema_manager.get_tool_returns(tool_name)


TOOL_DESCRIPTIONS = get_tool_descriptions()
TOOL_TYPE_MAPPING = _schema_manager.tool_type_mapping
