"""Toolkit for the ride-hailing domain."""

import json
import math
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from eff.domains.ridehailing.data_model import (
    Driver,
    RideOrder,
    RidehailingDB,
)
from eff.environment.toolkit import ToolKitBase, ToolType, is_tool
from eff.utils.utils import get_hash

# ---------------------------------------------------------------------------
# Driver generation constants (mirrors ridehailing/build_tasks.py)
# ---------------------------------------------------------------------------

_DRIVER_TAGS = [
    "本地通司机", "经验丰富", "车内宽敞", "后备箱大", "后座可放倒",
    "车龄新", "油车", "电车", "混动", "经常跑长途", "夜间接单较多",
    "驾驶专心", "驾驶平稳", "0投诉", "准时接送", "服务态度好",
    "无异味", "常开空调", "会英语", "帮拿行李", "路线用时短",
    "躲避拥堵", "高速优先", "更少附加费", "尽快上车", "可放音乐",
    "不吸烟", "有香薰", "可以充电", "男司机", "女司机",
    "耐心等待", "静音车厢", "车内安静", "讲普通话", "提供饮品",
    "车内整洁", "宠物接送", "提供无障碍服务", "甄选", "豪华车",
    "1年新车", "满分服务", "党员司机", "退役军人",
    "SUV", "MPV", "5座", "6座", "7座", "品牌车系",
]

# (品牌, 车型, 颜色列表, 品类标签, 座位数)
_DRIVER_POOL = [
    ("大众", "朗逸",      ["白", "银", "灰", "黑"],        "economy", 5),
    ("大众", "速腾",      ["白", "银", "黑", "蓝"],        "economy", 5),
    ("大众", "帕萨特",    ["黑", "白", "银", "灰"],        "premium", 5),
    ("大众", "途观L",     ["白", "黑", "灰", "银"],        "premium", 5),
    ("丰田", "卡罗拉",    ["白", "银", "红", "蓝", "灰"],  "economy", 5),
    ("丰田", "凯美瑞",    ["白", "黑", "银", "灰"],        "premium", 5),
    ("丰田", "汉兰达",    ["白", "黑", "灰", "银"],        "premium", 5),
    ("丰田", "埃尔法",    ["白", "黑", "银"],              "six_seat_luxury", 7),
    ("本田", "思域",      ["白", "红", "蓝", "银", "黑"],  "economy", 5),
    ("本田", "雅阁",      ["白", "银", "黑", "灰"],        "premium", 5),
    ("本田", "奥德赛",    ["白", "银", "灰"],              "six_seat", 6),
    ("日产", "轩逸",      ["白", "银", "灰", "红"],        "economy", 5),
    ("日产", "天籁",      ["白", "银", "黑"],              "premium", 5),
    ("日产", "奇骏",      ["白", "黑", "灰", "红"],        "premium", 5),
    ("别克", "英朗",      ["白", "银", "灰", "红"],        "economy", 5),
    ("别克", "君越",      ["白", "黑", "银"],              "premium", 5),
    ("别克", "GL8",       ["白", "黑", "银", "灰"],        "six_seat", 6),
    ("奔驰", "C级",       ["白", "黑", "银", "灰"],        "luxury",  5),
    ("奔驰", "E级",       ["黑", "白", "银"],              "luxury",  5),
    ("奔驰", "V级",       ["黑", "白", "银"],              "six_seat_luxury", 7),
    ("宝马", "3系",       ["白", "黑", "蓝", "灰"],        "luxury",  5),
    ("宝马", "5系",       ["黑", "白", "银", "蓝"],        "luxury",  5),
    ("宝马", "X5",        ["黑", "白", "灰"],              "luxury",  5),
    ("特斯拉", "Model 3", ["白", "红", "黑", "灰"],        "mid",     5),
    ("特斯拉", "Model Y", ["白", "红", "黑"],              "premium", 5),
    ("比亚迪", "汉",      ["白", "红", "黑", "蓝"],        "mid",     5),
    ("比亚迪", "宋Pro",   ["白", "蓝", "黑", "灰"],        "mid",     5),
    ("比亚迪", "唐",      ["白", "黑", "红"],              "premium", 5),
    ("理想", "L9",        ["白", "黑", "灰"],              "six_seat_luxury", 6),
    ("理想", "L8",        ["白", "黑", "灰", "银"],        "six_seat", 6),
    ("吉利", "帝豪",      ["白", "银", "蓝", "红"],        "economy", 5),
    ("长安", "逸动",      ["白", "银", "红", "灰"],        "economy", 5),
    ("荣威", "i5",        ["白", "银", "灰", "蓝"],        "economy", 5),
    ("起亚", "K3",        ["白", "银", "红", "黑"],        "economy", 5),
    ("现代", "伊兰特",    ["白", "银", "红", "蓝"],        "economy", 5),
    ("红旗", "H5",        ["白", "黑", "银", "红"],        "premium", 5),
    ("红旗", "H9",        ["黑", "白", "银"],              "luxury",  5),
]

_SURNAMES = [
    "王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
    "徐", "孙", "朱", "马", "胡", "郭", "林", "何", "高", "韩",
    "唐", "曹", "许", "邓", "肖", "冯", "曾", "程", "蔡", "彭",
]

_PLATE_PREFIXES = ["滴A", "滴B", "滴C", "新A", "新B", "新C", "滴D", "新D"]
_PLATE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXY"
_PLATE_DIGITS = "0123456789"

# 价格体系：特惠 < 快车 < 特快 < 出租车 < 优享 < 专车 < 六座专车 < 豪华车 < 六座豪华车
# (min_per_km, max_per_km, base_fee)
# 参数由两组真实数据回归拟合（6.3km / 18.5km），拟合残差 < 0.1 元：
#   6.3km:  特惠19.7  快车22.1  特快22.7  出租车22.9  专车39.2  六座专车57.1  豪华车107.5  六座豪华车169.9
#   18.5km: 特惠43.6  快车47.4  特快51.9  出租车63.5  专车79.8  六座专车116.9 豪华车219.0  六座豪华车≈346
_CAR_TYPE_PRICING = {
    "特惠":       (1.8,  2.1,  7.4),
    "快车":       (1.9,  2.2,  9.0),
    "特快":       (2.2,  2.6,  7.6),
    "出租车":     (2.8,  3.2, 13.0),  # 国内出租车: 起步价 13 元 (~3km), 续程 2.8-3.2 元/km
    "优享":       (2.5,  2.9, 13.5),  # 中端舒适：介于快车 / 专车之间（6.3km≈29.7, 18.5km≈63.5）
    "专车":       (3.1,  3.6, 18.2),
    "六座专车":   (4.6,  5.2, 26.2),
    "豪华车":     (8.5,  9.8, 49.9),
    "六座豪华车": (13.4, 15.5, 78.9),
}
# 价格阶梯（6.3km 基准）：
#   特惠 19.7 < 快车 22.1 < 特快 22.7 < 出租车 22.9 < 专车 39.2 < 六座专车 57.1 < 豪华车 107.5 < 六座豪华车 169.9

# 各车型对应的车辆品类（用于从 _DRIVER_POOL 筛选合适车辆）
_CAR_TYPE_POOL_MAP = {
    "特惠":       ["economy"],
    "快车":       ["economy", "mid"],
    "特快":       ["mid", "economy"],
    "出租车":     ["economy"],
    "优享":       ["mid", "premium"],   # 中端舒适：偏 mid 偶有 premium
    "专车":       ["premium"],
    "六座专车":   ["six_seat", "six_seat_luxury"],
    "豪华车":     ["luxury"],
    "六座豪华车": ["six_seat_luxury"],
}


def _rand_plate_no(rng: random.Random) -> str:
    """生成随机车牌号，格式：前缀·字母+4位字母数字。"""
    prefix = rng.choice(_PLATE_PREFIXES)
    mid = rng.choice(_PLATE_LETTERS)
    suffix = "".join(rng.choices(_PLATE_LETTERS + _PLATE_DIGITS, k=4))
    return f"{prefix}·{mid}{suffix}"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间球面距离（km）。"""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


_LEGAL_CAR_TYPES = frozenset(_CAR_TYPE_PRICING.keys())  # 合法车型集

# 常见非法车型 → 推荐合法车型的提示映射（仅在错误消息中暴露给 agent，不进 prompt）
# 用于 graceful error：agent 传入"优享/舒适型/商务/经济/性价比"等市场化品类时，
# tool 报错 + 给出推荐替代，让 agent 一轮重试即可选对。
_CAR_TYPE_ALIASES = {
    "舒享":       "优享 或 专车",
    "舒适":       "专车",
    "舒适型":     "专车",
    "经济":       "特惠 或 快车",
    "经济型":     "特惠 或 快车",
    "便宜":       "特惠（最便宜）",
    "性价比":     "特惠 或 快车",
    "高端":       "豪华车 或 专车",
    "商务":       "专车 或 豪华车",
    "商务型":     "专车",
    "豪华":       "豪华车",
    "礼宾":       "豪华车 或 专车",
    "尊享":       "豪华车 或 专车",
    "MPV":        "六座专车 或 六座豪华车",
    "mpv":        "六座专车 或 六座豪华车",
    "六座":       "六座专车 或 六座豪华车",
    "六座快车":   "六座专车（已下线六座快车，统一升级到六座专车）",
    "顺风车":     "特惠（最便宜的合法选项）",
    "拼车":       "特惠",
    "出行":       "快车（默认）",
}


def _legal_types_error_msg(illegal: list) -> str:
    """构造非法车型的错误消息，含推荐替代。"""
    legal_sorted = sorted(_LEGAL_CAR_TYPES)
    msg = f"Invalid car type(s): {illegal}. Legal types: {legal_sorted}."
    hints = []
    for it in illegal:
        for alias, suggest in _CAR_TYPE_ALIASES.items():
            if alias in it or it in alias:
                hints.append(f"'{it}' → 推荐：{suggest}")
                break
    if hints:
        msg += " 替代建议：" + "；".join(hints)
    return msg


def _estimate_fee(car_type: str, distance_km: float) -> float:
    """同车型同距离的确定性预估价：取 min/max per_km 的中点。
    保证 get_ride_estimate / match_drivers / 任何调用点对相同 (car_type, distance)
    返回完全一致的 estimate_fee（与路线/起终点保持线性可推）。
    """
    min_per_km, max_per_km, base_fee = _CAR_TYPE_PRICING.get(car_type, (1.0, 1.3, 7.0))
    per_km = (min_per_km + max_per_km) / 2.0
    return round(base_fee + per_km * distance_km, 1)


def _generate_random_drivers(car_types: List[str], trip_distance_km: float = 5.0,
                              seed_key: Optional[str] = None) -> List[dict]:
    """动态生成 10-20 个司机列表（基于 seed_key 稳定）。

    - **车型过滤**：非合法车型（如「拼车」「商务车」）会被丢弃，避免数据污染
    - **稳定性**：seed_key 用于固定随机数（同 task 同输入两次调用结果一致），
      默认基于 car_types + trip_distance_km，不同 agent 看到相同司机列表
    - 距用户：0.2-2.0 km 随机，按城区速度（15-25 km/h）换算到达分钟数
    - 预估费用：base_fee + trip_distance_km × price_per_km（按车型定价体系）
    - 优惠费用：0 到 min(5, fee×15%) 随机
    - 司机评分：80-100 随机
    - 标签：从 50 个标签中随机取 2-4 个
    """
    # 校验车型合法性：非法显式 raise，避免静默回退给假数据
    car_types = list(car_types or [])
    if not car_types:
        car_types = ["快车", "特惠", "专车"]  # 完全空时用默认组合
    illegal = [ct for ct in car_types if ct not in _LEGAL_CAR_TYPES]
    if illegal:
        raise ValueError(_legal_types_error_msg(illegal))

    # 基于 seed_key 固定随机数：同输入两次调用结果一致
    if seed_key is None:
        seed_key = f"{','.join(sorted(car_types))}|{trip_distance_km}"
    seed_int = int(get_hash(seed_key)[:8], 16)
    rng = random.Random(seed_int)
    n = rng.randint(10, 20)

    # 均匀分配车型配额，余数随机补齐
    n_per_type = max(1, n // len(car_types))
    type_list: List[str] = []
    for ct in car_types:
        type_list.extend([ct] * n_per_type)
    while len(type_list) < n:
        type_list.append(rng.choice(car_types))
    rng.shuffle(type_list)
    type_list = type_list[:n]

    drivers: List[dict] = []
    used_ids: set = set()

    for car_type in type_list:
        # 唯一 driver_id
        while True:
            did = f"D{rng.randint(10001, 99999)}"
            if did not in used_ids:
                used_ids.add(did)
                break

        # 按车型选取合适车辆
        categories = _CAR_TYPE_POOL_MAP.get(car_type, ["economy"])
        pool = [c for c in _DRIVER_POOL if c[3] in categories]
        if not pool:
            pool = _DRIVER_POOL
        brand, model, colors, _, _ = rng.choice(pool)
        color = rng.choice(colors)

        # 司机距用户距离：0.2-2.0 km；城区速度 15-25 km/h 换算时间
        driver_dist_km = round(rng.uniform(0.2, 2.0), 2)
        speed_kmh = rng.uniform(15.0, 25.0)
        eta_seconds = int(driver_dist_km / speed_kmh * 3600)
        distance_meters = int(driver_dist_km * 1000)

        # 预估费用：同车型同距离 → 确定性单价（不再每司机随机）
        estimate_fee = _estimate_fee(car_type, trip_distance_km)

        # 优惠费用：仍可按司机随机（不同司机促销不同），但上限收紧
        discount_fee = round(rng.uniform(0.0, min(5.0, estimate_fee * 0.15)), 1)

        # 司机评分 80-100，OTA/OEA
        recommend_score = rng.randint(80, 100)
        ota = round(rng.uniform(0.85, 0.99), 2)
        oea = round(rng.uniform(0.80, 0.98), 2)

        # 2-4 个个性化标签
        tags = rng.sample(_DRIVER_TAGS, rng.randint(2, 4))
        # 同义标签扩展：让"不吸烟" / "无烟"任一表述都能命中
        if "不吸烟" in tags and "无烟" not in tags:
            tags.append("无烟")

        drivers.append({
            "driver_id":       did,
            "driver_name":     rng.choice(_SURNAMES),
            "plate_no":        _rand_plate_no(rng),
            "brand_desc":      brand,
            "car_model":       model,
            "car_color":       color,
            "car_type":        car_type,
            "eta":             eta_seconds,
            "distance":        distance_meters,
            "estimate_fee":    estimate_fee,
            "discount_fee":    discount_fee,
            "recommend_score": recommend_score,
            "tags":            tags,
            "ota":             ota,
            "oea":             oea,
        })

    return drivers


def _inject_brand_drivers(brands: List[str], car_types: List[str], trip_distance_km: float,
                          seed_key: str, n: int = 2) -> List[dict]:
    """硬品牌过滤 0 命中时，从 _DRIVER_POOL 找匹配品牌的车，注入 n 辆司机。

    保证用户指定的品牌（如"红旗""比亚迪汉""奥迪"）即便随机采样没采到，
    也能强制返回 ≥1 辆该品牌车，避免 agent 因 0 候选无限重试。
    """
    seed_int = int(get_hash(seed_key)[:8], 16)
    rng = random.Random(seed_int)

    # 找匹配品牌的池条目
    matched_pool = []
    for entry in _DRIVER_POOL:
        b = entry[0]
        if any(target in b or b in target for target in brands):
            matched_pool.append(entry)
    if not matched_pool:
        return []

    drivers: List[dict] = []
    used_ids: set = set()
    for i in range(n):
        brand, model, colors, _, _ = rng.choice(matched_pool)
        # 选第一个用户指定的车型（如同时多车型，取第一个匹配 _CAR_TYPE_POOL_MAP 的）
        car_type = car_types[i % len(car_types)] if car_types else "快车"
        color = rng.choice(colors)

        while True:
            did = f"D{rng.randint(10001, 99999)}"
            if did not in used_ids:
                used_ids.add(did)
                break

        driver_dist_km = round(rng.uniform(0.2, 2.0), 2)
        speed_kmh = rng.uniform(15.0, 25.0)
        eta_seconds = int(driver_dist_km / speed_kmh * 3600)
        distance_meters = int(driver_dist_km * 1000)
        estimate_fee = _estimate_fee(car_type, trip_distance_km)
        discount_fee = round(rng.uniform(0.0, min(5.0, estimate_fee * 0.15)), 1)
        recommend_score = rng.randint(80, 100)
        ota = round(rng.uniform(0.85, 0.99), 2)
        oea = round(rng.uniform(0.80, 0.98), 2)

        tags = rng.sample(_DRIVER_TAGS, rng.randint(2, 4))
        if "不吸烟" in tags and "无烟" not in tags:
            tags.append("无烟")
        # 加上品牌相关标签，让 agent 能识别这是按品牌偏好挑的
        tags.append("品牌车系")

        drivers.append({
            "driver_id":       did,
            "driver_name":     rng.choice(_SURNAMES),
            "plate_no":        _rand_plate_no(rng),
            "brand_desc":      brand,
            "car_model":       model,
            "car_color":       color,
            "car_type":        car_type,
            "eta":             eta_seconds,
            "distance":        distance_meters,
            "estimate_fee":    estimate_fee,
            "discount_fee":    discount_fee,
            "recommend_score": recommend_score,
            "tags":            tags,
            "ota":             ota,
            "oea":             oea,
        })

    return drivers


class RidehailingTools(ToolKitBase):
    """All the tools for the ride-hailing domain."""

    db: RidehailingDB

    # 打车域不需要的公共工具，从 agent 可见列表中排除
    _EXCLUDED_TOOLS = frozenset({
        "get_date_holiday_info",
        "get_holiday_date",
        "get_user_historical_behaviors",
        "longitude_latitude_to_distance",
    })

    def __init__(self, db: RidehailingDB) -> None:
        super().__init__(db)

    @property
    def tools(self):
        return {name: func for name, func in super().tools.items() if name not in self._EXCLUDED_TOOLS}

    def _check_user(self, user_id: str) -> bool:
        if user_id != self.db.user_id:
            return False
        return True

    def _generate_order_id(self, user_id: str) -> str:
        timestamp = datetime.now().timestamp()
        return "RH" + get_hash(f"#RIDEHAILING#{user_id}{timestamp}")[:10]

    @is_tool(ToolType.GENERIC)
    def address_to_longitude_latitude(self, address: str) -> list:
        """根据地址获取经纬度。在打车域中会额外搜索 task 环境中的 POI 数据。"""
        assert address and address.strip(), "Address cannot be empty"
        address = address.strip()

        # 先从打车域环境数据中查找（start_poi / end_poi / location / home / company）
        candidates = []
        for poi in [self.db.start_poi, self.db.end_poi, self.db.home_address, self.db.company_address]:
            if poi and isinstance(poi, dict) and poi.get("lng") and poi.get("lat"):
                name = poi.get("displayname", "") or poi.get("address", "")
                if name:
                    candidates.append((name, poi["lng"], poi["lat"]))
        for loc in (self.db.location or []):
            if isinstance(loc, dict) and loc.get("longitude") and loc.get("latitude"):
                name = loc.get("address", "")
                if name:
                    candidates.append((name, loc["longitude"], loc["latitude"]))

        # 模糊匹配：任一候选的名称与查询地址有子串关系
        from difflib import SequenceMatcher
        best_match = None
        best_score = 0
        for name, lng, lat in candidates:
            # 子串匹配
            if address in name or name in address:
                return [lng, lat]
            # 相似度匹配
            score = SequenceMatcher(None, address, name).ratio()
            if score > best_score:
                best_score = score
                best_match = (lng, lat)
        if best_match and best_score >= 0.4:
            return list(best_match)

        # 回退到全局地址库
        return super().address_to_longitude_latitude(address)

    @is_tool(ToolType.READ)
    def search_pois(self, query: str, limit: int = 5) -> str:
        """搜索匹配查询关键词的所有 POI 候选，返回多个候选项以支持地址歧义消解。

        匹配优先级：
        1. task environment 中的具体 POI（start/end/home/company/location/history）
        2. 全城常见连锁品牌池（奶茶/咖啡/快餐/超市等品类匹配）
        3. 子串/相似度匹配兜底

        与 address_to_longitude_latitude 不同：本工具返回所有相似度足够高的候选，
        agent 可识别歧义并主动向用户澄清，而不是默默选第一个。
        """
        assert query and query.strip(), "Query cannot be empty"
        query = query.strip()

        # 收集所有 POI 候选
        candidates = []
        # 当前任务上下文 POI
        for label, poi in [
            ("start_poi", self.db.start_poi),
            ("end_poi", self.db.end_poi),
            ("home_address", self.db.home_address),
            ("company_address", self.db.company_address),
        ]:
            if poi and isinstance(poi, dict) and poi.get("lng") and poi.get("lat"):
                name = poi.get("displayname", "") or poi.get("address", "")
                if name:
                    candidates.append({
                        "name": name,
                        "address": poi.get("address", name),
                        "lng": poi["lng"], "lat": poi["lat"],
                        "source": label,
                    })
        # 任务环境的 POI 列表（兼容 dict 和 Location pydantic 对象）
        for loc in (self.db.location or []):
            if isinstance(loc, dict):
                lng_v = loc.get("longitude")
                lat_v = loc.get("latitude")
                addr = loc.get("address", "")
            else:
                # pydantic Location 对象（address/longitude/latitude 属性）
                lng_v = getattr(loc, "longitude", None)
                lat_v = getattr(loc, "latitude", None)
                addr = getattr(loc, "address", "")
            if lng_v is not None and lat_v is not None and addr:
                candidates.append({
                    "name": addr, "address": addr,
                    "lng": lng_v, "lat": lat_v,
                    "source": "location",
                })
        # 历史订单中的目的地
        for order in (self.db.history_orders or []):
            for pos in ("start_poi", "end_poi"):
                v = order.get(pos)
                if isinstance(v, str) and v:
                    candidates.append({
                        "name": v, "address": v,
                        "lng": None, "lat": None,
                        "source": f"history.{pos}",
                    })
        # task environment 的 nearby_pois 字段（与 nearby_search 共用）
        nearby = getattr(self.db, "nearby_pois", None) or {}
        for category, pois in (nearby.items() if isinstance(nearby, dict) else []):
            for p in pois or []:
                name = p.get("name") if isinstance(p, dict) else None
                if not name:
                    continue
                candidates.append({
                    "name": name, "address": name,
                    "lng": p.get("lng"), "lat": p.get("lat"),
                    "source": f"nearby_pois.{category}",
                })

        # 常见连锁品类 POI 池（按 query 关键词匹配类别后追加候选）
        # 用于 agent 在 task environment 没有显式 POI 时，仍能返回类别候选让 user 选
        BRAND_POOL = {
            "奶茶": ["喜茶", "奈雪的茶", "霸王茶姬", "CoCo都可", "一点点", "茶颜悦色"],
            "咖啡": ["星巴克", "瑞幸咖啡", "Manner Coffee", "%Arabica", "Tim Hortons", "Seesaw Coffee"],
            "快餐": ["麦当劳", "肯德基", "汉堡王", "必胜客"],
            "餐厅": ["海底捞", "西贝莜面村", "外婆家", "新白鹿"],
            "超市": ["盒马鲜生", "永辉超市", "大润发", "山姆会员店"],
            "商场": ["万达广场", "银泰百货", "大悦城", "万象城"],
            "医院": ["人民医院", "中山医院", "协和医院", "市第一医院"],
            "银行": ["中国银行", "工商银行", "建设银行", "招商银行"],
            "酒店": ["全季酒店", "汉庭酒店", "亚朵酒店", "锦江之星"],
            "花店": ["花点时间", "野兽派", "盒马花花", "Roseonly", "花加"],
            "便利店": ["7-Eleven", "全家", "罗森", "便利蜂"],
            "加油站": ["中石化", "中石油", "壳牌", "BP"],
            "地铁站": ["1号线站", "2号线站", "3号线站", "4号线站"],
            "地铁口": ["A出口", "B出口", "C出口", "D出口"],
        }
        # 用 city POI 字段推断参考坐标 + 城市名作为分店命名前缀
        ref_lng, ref_lat = None, None
        ref_landmark = ""  # 附近地标，用作分店名
        for poi_attr in ("start_poi","end_poi","home_address","company_address"):
            poi = getattr(self.db, poi_attr, None) or {}
            if isinstance(poi, dict) and poi.get("lng") and poi.get("lat"):
                ref_lng, ref_lat = poi["lng"], poi["lat"]
                ref_landmark = poi.get("displayname","") or poi.get("address","")
                break
        # 拿 city_name 作为城市前缀（直接读 db.city_name，旧版 db.environment 路径已弃用）
        city_name = getattr(self.db, "city_name", "") or ""

        # 分店区域名（基于参考地标抽取）：取地标的核心词（如「望京东园四区」→「望京」）
        def derive_district(landmark):
            for k in ["望京","三里屯","国贸","中关村","西单","朝阳","海淀","CBD","机场","火车站",
                      "陆家嘴","张江","虹桥","徐汇","静安","浦东","古北",
                      "天河","越秀","海珠","白云","珠江新城",
                      "福田","南山","罗湖","宝安","龙华","科技园",
                      "西湖","钱江","萧山","余杭","拱墅",
                      "高新","曲江","雁塔","钟楼","咸阳",
                      "锦江","春熙路","天府","双流","高新区",
                      "光谷","汉口","武昌","江汉",
                      "新街口","河西","江宁","仙林","鼓楼",
                      "软件园","翔安","海沧","集美","思明"]:
                if k in landmark:
                    return k
            return None
        district = derive_district(ref_landmark) or "市区"

        # 机场航站楼歧义：query 含"机场"且未指定 T1/T2/T3 → 返回多个航站楼候选
        if "机场" in query and not any(t in query for t in ["T1", "T2", "T3", "t1", "t2", "t3", "1号", "2号", "3号"]):
            # 城市机场字典
            city_airport = {
                "北京": "首都国际机场", "上海": "浦东国际机场",
                "广州": "白云国际机场", "深圳": "宝安国际机场",
                "杭州": "萧山国际机场", "成都": "天府国际机场",
                "西安": "咸阳国际机场", "武汉": "天河国际机场",
                "南京": "禄口国际机场", "厦门": "高崎国际机场",
                "石家庄": "正定国际机场",
            }
            # query 中的具体机场关键词 → 城市映射（防跨城静默匹配）
            query_airport_city = {
                "首都": "北京", "浦东": "上海", "虹桥": "上海",
                "白云": "广州", "宝安": "深圳", "萧山": "杭州",
                "天府": "成都", "双流": "成都", "咸阳": "西安",
                "天河": "武汉", "禄口": "南京", "高崎": "厦门",
                "正定": "石家庄",
            }
            # 优先按 query 中的具体机场名识别目标城市，再 fallback 到 ctx city
            airport_name = None
            for keyword, target_city in query_airport_city.items():
                if keyword in query:
                    airport_name = city_airport.get(target_city)
                    if airport_name:
                        break
            if not airport_name:
                for c, an in city_airport.items():
                    if c in city_name or c in (ref_landmark or ""):
                        airport_name = an
                        break
            airport_name = airport_name or "机场"
            # 各机场实际航站楼配置（与真实世界一致）
            airport_terminals = {
                "首都国际机场": ["T1", "T2", "T3"],   # 北京首都
                "浦东国际机场": ["T1", "T2"],          # 上海浦东
                "白云国际机场": ["T1", "T2"],          # 广州白云
                "宝安国际机场": ["T3"],                # 深圳宝安（只有 T3，T1/T2 已停用/不存在）
                "萧山国际机场": ["T1", "T2", "T3"],   # 杭州萧山
                "天府国际机场": ["T1", "T2"],          # 成都天府
                "咸阳国际机场": ["T2", "T3", "T5"],   # 西安咸阳
                "天河国际机场": ["T1", "T2", "T3"],   # 武汉天河
                "禄口国际机场": ["T1", "T2"],          # 南京禄口
                "高崎国际机场": ["T3", "T4"],          # 厦门高崎（只有 T3 T4）
                "正定国际机场": ["T1", "T2"],          # 石家庄正定
            }
            # 各机场真实坐标（航站楼小幅偏移 < 1km，整体接近真实位置）
            airport_real_coords = {
                "首都国际机场": (116.587, 40.080),
                "浦东国际机场": (121.806, 31.151),
                "白云国际机场": (113.297, 23.395),
                "宝安国际机场": (113.811, 22.640),
                "萧山国际机场": (120.435, 30.235),
                "天府国际机场": (104.443, 30.310),
                "咸阳国际机场": (108.751, 34.452),
                "天河国际机场": (114.213, 30.781),
                "禄口国际机场": (118.862, 31.745),
                "高崎国际机场": (118.130, 24.546),
                "正定国际机场": (114.696, 38.281),
            }
            terminals = airport_terminals.get(airport_name, ["T1", "T2", "T3"])
            base_lng, base_lat = airport_real_coords.get(airport_name, (None, None))
            for i, t in enumerate(terminals):
                # 用真实机场坐标 + 航站楼间 ~500m 小幅偏移，避免占位坐标误导后续工具
                if base_lng is not None:
                    lng = round(base_lng + i * 0.005, 4)
                    lat = round(base_lat + i * 0.003, 4)
                else:
                    # fallback: 起点偏移（机场未列入字典）
                    offset = 0.01 * (i + 1)
                    lng = round(ref_lng + offset, 4) if ref_lng else None
                    lat = round(ref_lat + offset * 0.6, 4) if ref_lat else None
                name = f"{airport_name}{t}航站楼"
                candidates.append({
                    "name": name, "address": name,
                    "lng": lng, "lat": lat,
                    "source": "chain.airport",
                })

        # 火车站/高铁站歧义：query 含"火车站/高铁站/车站"且未指定具体站名 → 按城市返回多个站候选
        _STATION_TRIGGER = ["火车站", "高铁站", "动车站"]
        _STATION_GENERIC = "车站"  # 兜底（避免误命中"地铁站"）
        is_station_query = (
            any(k in query for k in _STATION_TRIGGER)
            or (_STATION_GENERIC in query and "地铁" not in query)
        )
        # 排除已含具体站名的查询（如"南京南站"）
        already_specific = any(s in query for s in [
            "南站", "北站", "东站", "西站", "虹桥", "汉口", "武昌",
        ])
        if is_station_query and not already_specific:
            city_stations = {
                "北京": ["北京站", "北京西站", "北京南站", "北京北站"],
                "上海": ["上海站", "上海虹桥站", "上海南站"],
                "广州": ["广州站", "广州南站", "广州东站"],
                "深圳": ["深圳站", "深圳北站", "深圳东站"],
                "杭州": ["杭州站", "杭州东站", "杭州西站"],
                "成都": ["成都站", "成都东站", "成都南站"],
                "西安": ["西安站", "西安北站", "西安南站"],
                "武汉": ["武汉站", "武昌站", "汉口站"],
                "南京": ["南京站", "南京南站"],
                "厦门": ["厦门站", "厦门北站"],
            }
            stations = None
            for c, sl in city_stations.items():
                if c in city_name or c in (ref_landmark or ""):
                    stations = sl
                    break
            stations = stations or ["火车站（市中心）", "高铁站（新城区）"]
            for i, s in enumerate(stations):
                offset = 0.012 * (i + 1)
                lng = round(ref_lng + offset, 4) if ref_lng else None
                lat = round(ref_lat - offset * 0.5, 4) if ref_lat else None
                candidates.append({
                    "name": s, "address": s,
                    "lng": lng, "lat": lat,
                    "source": "chain.station",
                })

        # 区分两种 query：
        #   (a) 具体品牌（"星巴克"/"麦当劳"/"瑞幸咖啡"）→ 只返回该品牌的多个分店；
        #   (b) 通用品类（"咖啡店"/"奶茶"/"酒店"）       → 返回该品类下多个品牌的代表分店。
        regions = [district, "周边商圈", "近地铁站", "附近购物中心", "市中心", "近火车站"]
        # 若 query 完全等于某个品类词（"酒店"/"咖啡"），跳过 matched_brand 检测——
        # 避免 query="酒店" 因 "酒店" 是 "全季酒店" 的子串而被误判为单品牌查询。
        # 注意：仅完全相等时跳过，"瑞幸咖啡"/"奶茶店" 这类不算品类词等价。
        matched_brand = None
        if query not in BRAND_POOL:
            # 提取连续公共子串：兼容空格分隔/带修饰词的 query，如"望京 西贝莜面"应能匹中"西贝莜面村"
            def _lcs_len(s1: str, s2: str) -> int:
                if not s1 or not s2:
                    return 0
                m, n = len(s1), len(s2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                longest = 0
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if s1[i - 1] == s2[j - 1]:
                            dp[i][j] = dp[i - 1][j - 1] + 1
                            if dp[i][j] > longest:
                                longest = dp[i][j]
                return longest

            for cat, brands in BRAND_POOL.items():
                for b in brands:
                    # 双向匹配：query 含 b 或 b 含 query
                    if b in query or (len(query) >= 2 and query in b):
                        matched_brand = b
                        break
                    # LCS 兜底：query 与 b 的最长公共子串 ≥ 3 字 且 ≥ len(b)*0.6
                    # 例：query="望京 西贝莜面" vs b="西贝莜面村" → LCS="西贝莜面"(4) ≥ 0.6*5=3 ✓
                    if len(b) >= 3:
                        lcs = _lcs_len(query, b)
                        if lcs >= 3 and lcs >= len(b) * 0.6:
                            matched_brand = b
                            break
                if matched_brand:
                    break

        if matched_brand:
            # 单品牌多分店（保证返回结果与 query 相关性 >95%）
            for i, region in enumerate(regions):
                offset = (i + 1) * 0.005
                lng = round(ref_lng + offset, 4) if ref_lng else None
                lat = round(ref_lat + offset * 0.6, 4) if ref_lat else None
                name = f"{matched_brand}（{region}店）"
                candidates.append({
                    "name": name, "address": name,
                    "lng": lng, "lat": lat,
                    "source": f"chain.brand.{matched_brand}",
                })
        else:
            # 通用品类：每个品牌出一个代表分店
            # 仅当 query 主体就是品类词时才触发（去掉品类词剩余 ≤ 3 字）。
            # 例：query="奶茶店" / "找个奶茶" → 触发；query="昆明丽瑞德酒店" → 剩 5 字，不触发，
            # 避免把"含品类后缀的具体店名"误判为品类查询而召回无关品牌池。
            for cat, brands in BRAND_POOL.items():
                if cat in query and (len(query) - len(cat)) <= 3:
                    for i, b in enumerate(brands):
                        region = regions[i % len(regions)]
                        offset = (i + 1) * 0.005
                        lng = round(ref_lng + offset, 4) if ref_lng else None
                        lat = round(ref_lat + offset * 0.6, 4) if ref_lat else None
                        name = f"{b}（{region}店）"
                        candidates.append({
                            "name": name, "address": name,
                            "lng": lng, "lat": lat,
                            "source": f"chain.category.{cat}",
                        })
                    break  # 单品类，找到就够

        # 评分匹配：子串优先，相似度作为兜底
        from difflib import SequenceMatcher

        def _longest_common_substring(s1: str, s2: str) -> str:
            """返回 s1 / s2 的最长公共子串（连续）；空串则空。"""
            if not s1 or not s2:
                return ""
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            longest, end = 0, 0
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        if dp[i][j] > longest:
                            longest = dp[i][j]
                            end = i
            return s1[end - longest:end]

        def _name_similarity(query: str, name: str) -> float:
            """中文 POI 名相似度。规则：
            - 双向完全包含 → 1.0（query 是 name 的连续子串或反之）
            - 最长公共子串（LCS）≥ max(3, len(query)//2) → 至少 0.92（核心词命中）
            - LCS ≥ 3 → 至少 0.78（部分命中）
            - 其他 → SequenceMatcher.ratio() 裸值
            """
            if not query or not name:
                return 0.0
            if query in name or name in query:
                return 1.0
            base = SequenceMatcher(None, query, name).ratio()
            lcs = _longest_common_substring(query, name)
            lcs_len = len(lcs)
            if lcs_len >= max(3, len(query) // 2):
                return max(base, 0.92)
            if lcs_len >= 3:
                return max(base, 0.78)
            return base

        seen_names = set()
        scored = []
        for c in candidates:
            name = c["name"]
            if name in seen_names:
                continue
            seen_names.add(name)
            # 连锁池来源（chain.*）：直接给高分，不做相似度过滤（用户问"奶茶店"应该看到所有奶茶分店）
            src = c.get("source", "")
            if src.startswith("chain.brand.") or src in ("chain.airport", "chain.station"):
                score = 0.95  # 同品牌不同分店 / 同机场不同航站楼 / 同城多火车站，与 query 高度相关
                # station 类型优先：query 含"高铁"时，name 含"高铁"的提分；query 含"火车"时同理
                if src == "chain.station":
                    if "高铁" in query and "高铁" in name:
                        score = 0.99
                    elif "动车" in query and ("高铁" in name or "动车" in name):
                        score = 0.98
                    elif "火车" in query and "火车" in name:
                        score = 0.99
            elif src.startswith("chain.category."):
                # 品类候选：仅当 query 主体即品类词（如 "医院"/"咖啡"）时统一给 0.95；
                # 若 query 含修饰词（如 "儿童医院"），通用品牌不一定相关 → 用实际相似度，
                # 避免 "人民医院"/"协和医院" 在 query="儿童医院" 下被误标 0.95。
                cat = src.split(".", 2)[2] if src.count(".") >= 2 else ""
                if cat and query == cat:
                    score = 0.95
                else:
                    # 去掉 region 后缀算品牌名相似度（"人民医院（国贸店）" → "人民医院"）
                    bare_name = name.split("（")[0] if "（" in name else name
                    score = _name_similarity(query, bare_name)
            else:
                # task 环境内 POI / history POI / 其他：完整相似度计算（含 LCS 奖励）
                score = _name_similarity(query, name)
            if score >= 0.4:
                scored.append((score, c))
        scored.sort(key=lambda x: -x[0])

        # 少结果兜底：scored < 3 且 query 含某品类关键词时，补充 (3 - len) 个该品类
        # 代表分店作为"参考候选"（低分 0.55），让 agent 至少有多家可对比。
        # 与"召回侧收紧"逻辑不冲突 —— 仅在主匹配数 < 3 时触发，且作为参考而非主排序。
        MIN_RESULTS = 3
        if len(scored) < MIN_RESULTS:
            matched_cat = None
            for cat in BRAND_POOL.keys():
                if cat in query:
                    matched_cat = cat
                    break
            if matched_cat:
                existing_names = {c['name'] for _, c in scored}
                brands = BRAND_POOL[matched_cat]
                for i, b in enumerate(brands):
                    if len(scored) >= MIN_RESULTS:
                        break
                    region = regions[i % len(regions)]
                    offset = (i + 10) * 0.005
                    lng = round(ref_lng + offset, 4) if ref_lng else None
                    lat = round(ref_lat + offset * 0.6, 4) if ref_lat else None
                    name = f"{b}（{region}店）"
                    if name in existing_names:
                        continue
                    scored.append((0.55, {
                        "name": name, "address": name,
                        "lng": lng, "lat": lat,
                        "source": f"chain.category.{matched_cat}",
                    }))
                    existing_names.add(name)

        if not scored:
            return f"未找到与 '{query}' 匹配的 POI 候选。建议向用户确认更具体的地点名称。"

        top = scored[: max(1, limit)]

        # 持久化召回结果到 db.location，后续工具（get_trip_duration_estimate /
        # get_ride_estimate / create_ride_order 等）通过 _lookup_coords 可访问。
        # 否则 search_pois 临时生成的 chain.brand / chain.airport 等虚拟分店
        # 在后续 via_addresses / start/end 引用时会查不到坐标，被静默 skip。
        def _get_addr(loc):
            return loc.get('address') if isinstance(loc, dict) else getattr(loc, 'address', None)
        existing = {_get_addr(loc) for loc in (self.db.location or []) if _get_addr(loc)}
        if self.db.location is None:
            self.db.location = []
        for _, c in top:
            name = c.get('name')
            lng, lat = c.get('lng'), c.get('lat')
            if name and lng is not None and lat is not None and name not in existing:
                self.db.location.append({'address': name, 'longitude': lng, 'latitude': lat})
                existing.add(name)

        lines = [f"找到 {len(top)} 个匹配 '{query}' 的 POI 候选："]
        for i, (score, c) in enumerate(top, 1):
            coord = f"经纬度({c['lng']}, {c['lat']})" if c["lng"] is not None else "（坐标未知，需进一步定位）"
            lines.append(f"[{i}] {c['name']} | {coord} | 来源: {c['source']} | 相似度: {score:.2f}")
        # 不附加"建议向用户确认"等行为提示：候选的相似度分数已客观呈现，
        # 由 agent 自行判断"明显匹配（直接采用）"还是"真歧义（需澄清）"。
        # 附加该提示会与评测 rubric（dominant match 应直接采用）冲突，
        # 等于诱导 agent 做出被惩罚的行为，污染该项能力的区分度。
        return "\n".join(lines)

    @is_tool(ToolType.READ)
    def get_user_all_orders(self) -> str:
        """获取用户所有订单信息"""
        # 打车域：返回历史订单 + 当前订单
        parts = []
        # 历史订单
        if self.db.history_orders:
            parts.append(f"历史打车订单（{len(self.db.history_orders)}条）：")
            for o in self.db.history_orders:
                parts.append(
                    f"  {o.get('order_id','?')} | {o.get('car_type','?')} | "
                    f"{o.get('start_poi','?')}→{o.get('end_poi','?')} | "
                    f"¥{o.get('estimate_fee',0)} | {o.get('status','?')}"
                )
        # 当前订单
        if self.db.ride_orders:
            parts.append(f"\n当前打车订单（{len(self.db.ride_orders)}条）：")
            for oid, order in self.db.ride_orders.items():
                parts.append(
                    f"  {oid} | {order.get('car_type','?')} | "
                    f"{(order.get('start_poi') or {}).get('displayname','?')}→"
                    f"{(order.get('end_poi') or {}).get('displayname','?')} | "
                    f"状态: {order.get('status','?')}"
                )
        if not parts:
            return "用户当前没有任何打车订单记录"
        return "\n".join(parts)

    @is_tool(ToolType.READ)
    def get_nearby(self, longitude: float, latitude: float, range: float) -> str:
        """搜索指定坐标附近的地点"""
        # 打车域：从环境 POI 中搜索附近地点
        results = []
        pois = []
        for poi in [self.db.start_poi, self.db.end_poi, self.db.home_address, self.db.company_address]:
            if poi and isinstance(poi, dict) and poi.get("lng") and poi.get("lat"):
                name = poi.get("displayname", "") or poi.get("address", "")
                if name:
                    pois.append((name, poi["lng"], poi["lat"]))
        for loc in (self.db.location or []):
            if isinstance(loc, dict) and loc.get("longitude") and loc.get("latitude"):
                name = loc.get("address", "")
                if name:
                    pois.append((name, loc["longitude"], loc["latitude"]))

        for name, lng, lat in pois:
            dist = _haversine_km(latitude, longitude, lat, lng)
            if dist <= range:
                results.append(f"{name}（距离{dist:.1f}km）")

        if not results:
            return "附近未找到相关地点"
        return "\n".join(results)

    @is_tool(ToolType.READ)
    def get_home_address(self, user_id: str) -> str:
        """获取用户家地址"""
        assert user_id, "User ID cannot be empty"
        assert self._check_user(user_id), "User ID does not match"
        if self.db.home_address is None:
            return "用户未设置家地址"
        return json.dumps(self.db.home_address, ensure_ascii=False)

    @is_tool(ToolType.READ)
    def get_company_address(self, user_id: str) -> str:
        """获取用户公司地址"""
        assert user_id, "User ID cannot be empty"
        assert self._check_user(user_id), "User ID does not match"
        if self.db.company_address is None:
            return "用户未设置公司地址"
        return json.dumps(self.db.company_address, ensure_ascii=False)

    @is_tool(ToolType.WRITE)
    def select_car_types(self, car_types: List[str]) -> str:
        """让用户选择车型"""
        assert car_types, "Car types cannot be empty"
        valid_types = self.db.available_car_types or ["快车", "专车", "出租车"]
        for ct in car_types:
            assert ct in valid_types, f"Invalid car type: {ct}. Available: {valid_types}"
        self.db.selected_car_types = car_types
        return f"已选择车型: {', '.join(car_types)}"

    @is_tool(ToolType.READ)
    def get_available_car_types(self) -> str:
        """获取可用车型列表"""
        car_types = self.db.available_car_types or ["快车", "专车", "出租车"]
        return json.dumps(car_types, ensure_ascii=False)

    @is_tool(ToolType.READ)
    def match_drivers(
        self,
        user_id: str,
        start_address: str,
        end_address: str,
        car_types: Optional[List[str]] = None,
        car_brands: Optional[List[str]] = None,
        via_addresses: Optional[List[str]] = None,
    ) -> str:
        """匹配司机，返回可选司机列表。

        参数：
          - user_id: 用户 ID
          - start_address / end_address: 起点 / 终点
          - car_types: 可选，指定车型筛选（如 ["六座专车","专车"]）
          - car_brands: 可选，指定车辆品牌筛选（如 ["奔驰","宝马","奥迪","特斯拉"]）。
            agent 应支持用户偏好；若品牌池中无完全匹配的，会按相近原则放宽匹配并返回。
          - via_addresses: 可选，途经点列表（按经过顺序）。仅用于按累计行程算预估价
            （estimate_fee），不影响司机筛选（司机基于起点附近匹配）。
        """
        assert user_id, "User ID cannot be empty"
        assert self._check_user(user_id), "User ID does not match"
        assert start_address, "Start address cannot be empty"
        assert end_address, "End address cannot be empty"

        selected_types = car_types or self.db.selected_car_types or ["快车", "专车", "出租车"]
        # 校验车型合法性：非法车型显式报错，避免静默回退掩盖 agent 错误
        illegal = [ct for ct in selected_types if ct not in _LEGAL_CAR_TYPES]
        if illegal:
            raise ValueError(_legal_types_error_msg(illegal))

        # 优先用 agent 实际传入的 start/end_address 计算距离（含途经点累计距离）
        trip_distance_km = self._compute_trip_distance(start_address, end_address, via_addresses)

        brand_note = ""

        # === 静态 driver_pool 模式（task.environment.available_drivers 非空时启用）===
        # 设计目标：跨模型完全一致 + task 设计者可精准埋点（如"考察女司机"必有 2 辆女司机）。
        # 与随机模式的关键差异：
        #   - 司机基础属性（id/name/plate/brand/model/tags/score/distance/eta）freeze 在 pool 里
        #   - trip_distance/estimate_fee 仍随 agent 实际查询动态算
        #   - discount_fee = round(estimate_fee × discount_ratio, 1)，ratio freeze 在 pool 里
        #   - filter 后空池 → 不 fallback 到 random，返回空 + 文案（维持"无 SUV → agent 妥协"考察意图）
        #   - 不再调 _inject_brand_drivers（task 已显式埋点）
        if self.db.available_drivers:
            pool = list(self.db.available_drivers)
            # 1) car_types 筛选
            matched = [d for d in pool if d.get("car_type") in selected_types]
            if not matched:
                self._last_matched_drivers = []
                return (
                    f"未匹配到符合车型 {selected_types} 的司机。"
                    f"当前车队覆盖车型: {sorted(set(d.get('car_type') for d in pool))}。"
                    f"建议向用户告知并推荐合理替代车型。"
                )
            # 2) car_brands 筛选（可选）
            if car_brands:
                normalized_brands = [b.strip() for b in car_brands if b.strip()]
                brand_filtered = [d for d in matched
                                  if any(b in d.get("brand_desc","") for b in normalized_brands)]
                if brand_filtered:
                    matched = brand_filtered
                    brand_note = f"（已按品牌偏好 {normalized_brands} 筛选）"
                else:
                    self._last_matched_drivers = []
                    return (
                        f"未匹配到品牌为 {normalized_brands} 的司机（当前车型 {selected_types} 下）。"
                        f"建议向用户说明该品牌暂无可用，并推荐相近车型/由用户决定是否调整偏好。"
                    )
            # 2.5) start_area 过滤（多段行程：不同起点附近是不同的司机）
            # 设计：司机可带可选 start_area（接驾区域，如"希尔顿"/"德基广场"）。
            #   - 池中存在任一带 start_area 的司机时启用本过滤；
            #   - 带 start_area 的司机仅在其区域与本次 start_address 匹配（任一为另一方子串）时出现；
            #   - 不带 start_area 的司机视为全局可用（任何起点都出现，向后兼容）；
            #   - 若过滤后为空（起点未命中任何区域），回退到不过滤，避免返回空列表。
            # 池中无任何 start_area 时，本段完全跳过 → 老 task 行为不变。
            if any(d.get("start_area") for d in pool):
                def _area_match(area: str, addr: str) -> bool:
                    a = (area or "").strip()
                    s = (addr or "").strip()
                    return bool(a) and bool(s) and (a in s or s in a)
                area_matched = [
                    d for d in matched
                    if not d.get("start_area") or _area_match(d.get("start_area"), start_address)
                ]
                if area_matched:
                    matched = area_matched

            # 3) 动态算 trip-related 字段
            # distance/eta：按 (driver_id + start_address) 做稳定扰动，让不同起点
            # 返回的"司机离你多远"不同（同一段内多次调用结果稳定）。
            # 司机基础属性（id/name/car/score/tags）仍 freeze。
            import hashlib
            def _perturb(driver_id: str, base: int) -> int:
                if not base or base <= 0:
                    return base
                seed = hashlib.md5(f"{driver_id}|{start_address}".encode()).hexdigest()
                h = int(seed[:8], 16) / 0xFFFFFFFF  # [0, 1)
                factor = 0.5 + h * 1.5  # 0.5x ~ 2.0x，司机离起点距离的合理范围
                return int(base * factor)

            enriched = []
            for d in matched:
                ct = d.get("car_type")
                fee = _estimate_fee(ct, trip_distance_km)
                ratio = float(d.get("discount_ratio", 0.0) or 0.0)
                discount = round(fee * ratio, 1)
                drv = dict(d)  # shallow copy
                drv["estimate_fee"] = fee
                drv["discount_fee"] = discount
                drv.pop("discount_ratio", None)
                drv.pop("start_area", None)  # 非 Driver 字段，构造前剔除
                # 兼容 Driver model 字段：eta/distance（秒/米）+ 按起点扰动
                base_dist = drv.pop("driver_distance_meters", drv.get("distance", 1500))
                base_eta = drv.pop("eta_seconds", drv.get("eta", 240))
                drv["distance"] = _perturb(d.get("driver_id",""), int(base_dist))
                drv["eta"] = _perturb(d.get("driver_id",""), int(base_eta))
                enriched.append(drv)
            # 重新按 eta 升序排（让"最近的"司机排在前面）
            enriched.sort(key=lambda x: x.get("eta", 9999))
            matched = enriched
            self._last_matched_drivers = matched
        else:
            # === Fallback：旧的随机生成路径（向后兼容空 pool 的老 task / r8/r9 baseline）===
            # 同 task 同输入下两模型应看到相同司机列表 → 用 user_id+start+end+车型 做 seed
            seed_key = f"match_drivers|{user_id}|{start_address}|{end_address}|{','.join(sorted(selected_types))}|brands={','.join(sorted(car_brands or []))}"
            matched = _generate_random_drivers(selected_types, trip_distance_km, seed_key=seed_key)

            # 品牌过滤（如果 user 指定了 car_brands，按品牌名做包含匹配；非包含的剔除，但保留至少 3 个）
            if car_brands:
                normalized_brands = [b.strip() for b in car_brands if b.strip()]
                filtered = [d for d in matched
                            if any(b in d.get("brand_desc","") for b in normalized_brands)]
                # 检查 _DRIVER_POOL 是否有该品牌车（决定能否硬注入）
                pool_brands = {b for b in (e[0] for e in _DRIVER_POOL)}
                available_in_pool = [b for b in normalized_brands
                                     if any(b in pb or pb in b for pb in pool_brands)]
                if len(filtered) >= 3:
                    matched = filtered
                    brand_note = f"（已按品牌偏好 {normalized_brands} 筛选）"
                elif filtered:
                    matched = filtered + [d for d in matched if d not in filtered][:max(0, 3-len(filtered))]
                    brand_note = f"（品牌偏好 {normalized_brands} 匹配仅 {len(filtered)} 个，已补充近似司机至 ≥3）"
                elif available_in_pool:
                    # 池里确实有该品牌车，但本次随机采样没采到 → 强制注入 ≥2 辆该品牌车
                    injected = _inject_brand_drivers(
                        available_in_pool, selected_types, trip_distance_km,
                        seed_key=seed_key + f"|inject={','.join(available_in_pool)}",
                        n=2,
                    )
                    matched = injected + matched[:max(0, 5 - len(injected))]
                    brand_note = f"（品牌偏好 {normalized_brands} 本次随机未采到，已从车队中匹配 {len(injected)} 辆该品牌车型）"
                else:
                    brand_note = f"（品牌偏好 {normalized_brands} 无完全匹配，按车型返回近似车辆）"

            # 缓存本次结果，供后续按序号/ID/姓名选择司机
            self._last_matched_drivers = matched

        lines = [f"为您匹配到{len(matched)}位司机{brand_note}："]
        for i, d in enumerate(matched, 1):
            driver = Driver(**d)
            dist_km = round(driver.distance / 1000, 2)
            eta_min = max(1, round(driver.eta / 60))
            lines.append(
                f"[{i}] driver_id: {driver.driver_id} | "
                f"{driver.driver_name}师傅 | {driver.car_type} | "
                f"{driver.brand_desc}·{driver.car_model}({driver.car_color}) | "
                f"车牌: {driver.plate_no} | "
                f"距你{dist_km}km·约{eta_min}分钟 | "
                f"预估: ¥{driver.estimate_fee} (优惠¥{driver.discount_fee}) | "
                f"司机评分: {driver.recommend_score} | "
                f"标签: {','.join(driver.tags)}"
            )
        return "\n".join(lines)

    def _effective_poi(self, db_poi: Optional[dict], fallback_address: str) -> dict:
        """根据 agent 传入的地址名构造结构化 POI。
        关键约束：返回的 POI **必须使用 db 中的标准 displayname**（而非 agent 传入的别名），
        以保证下单后 ride_orders.start_poi.displayname 始终为标准名，evaluator 可正确匹配。

        优先级：
        1. db_poi 的 displayname/address 与 fallback_address 匹配 → 用 db_poi（含标准 displayname + 坐标）
        2. 在 db 其他 POI 字段中能反查到 → **返回该 POI 的标准 displayname**（不用 fallback_address）
        3. 都不匹配（agent 用了未注册地址）→ 用 fallback_address 新建（无坐标）
        """
        if not fallback_address:
            return db_poi or {}
        # 1) db_poi 自己匹配
        if isinstance(db_poi, dict):
            name = db_poi.get("displayname") or db_poi.get("address") or ""
            if name == fallback_address or fallback_address in name or name in fallback_address:
                return db_poi
        # 2) 反查 db 其他字段（命中则返回标准 displayname，不是 agent 传入的别名）
        standard = self._lookup_full_poi(fallback_address)
        if standard:
            return standard
        # 3) fallback：用 address 字符串构造（无坐标）
        return {"displayname": fallback_address, "address": fallback_address}

    def _iter_poi_candidates(self):
        """遍历 db 中所有 POI 候选，返回 (displayname, address, lng, lat) 元组列表。
        兼容 dict 与 pydantic BaseModel。
        """
        def _get(obj, *keys):
            if isinstance(obj, dict):
                for k in keys:
                    v = obj.get(k)
                    if v is not None:
                        return v
            else:
                for k in keys:
                    v = getattr(obj, k, None)
                    if v is not None:
                        return v
            return None

        cands = []
        for poi_attr in ("start_poi", "end_poi", "home_address", "company_address"):
            poi = getattr(self.db, poi_attr, None)
            if poi is None:
                continue
            displayname = _get(poi, "displayname", "address")
            address = _get(poi, "address", "displayname")
            lng = _get(poi, "lng", "longitude")
            lat = _get(poi, "lat", "latitude")
            if displayname and lng is not None and lat is not None:
                cands.append((displayname, address or displayname, lng, lat))
        for loc in (self.db.location or []):
            if loc is None:
                continue
            displayname = _get(loc, "address", "displayname")
            address = _get(loc, "address", "displayname")
            lng = _get(loc, "longitude", "lng")
            lat = _get(loc, "latitude", "lat")
            if displayname and lng is not None and lat is not None:
                cands.append((displayname, address or displayname, lng, lat))
        return cands

    def _lookup_full_poi(self, address: str) -> Optional[dict]:
        """根据 agent 输入的地址（可能是别名/缩写）反查 db 中的标准 POI dict。
        返回 {displayname, address, lng, lat} —— displayname 始终是 db 标准名，不是 agent 传入的别名。
        匹配策略：精确相等 → 子串包含。
        """
        if not address:
            return None
        candidates = self._iter_poi_candidates()
        # 1) 精确相等（displayname 或 address）
        for dn, addr, lng, lat in candidates:
            if dn == address or addr == address:
                return {"displayname": dn, "address": addr, "lng": lng, "lat": lat}
        # 2) 子串包含（双向）
        for dn, addr, lng, lat in candidates:
            if address in dn or dn in address or address in addr or addr in address:
                return {"displayname": dn, "address": addr, "lng": lng, "lat": lat}
        return None

    def _lookup_coords(self, address: str) -> Optional[tuple]:
        """根据地址名查找坐标 (lng, lat)。保持向后兼容；底层走 _lookup_full_poi。"""
        poi = self._lookup_full_poi(address)
        if poi:
            return (poi["lng"], poi["lat"])
        return None

    def _compute_trip_distance(self, start_address: str, end_address: str,
                                via_addresses: Optional[List[str]] = None) -> float:
        """计算起点 → [途经点] → 终点的总公里数。
        优先按地址查 db 真实坐标；失败时退回 db.start_poi/end_poi 的直线距离。
        """
        try:
            points = []
            for addr in [start_address] + list(via_addresses or []) + [end_address]:
                xy = self._lookup_coords(addr)
                if xy is not None:
                    points.append(xy)
            if len(points) >= 2:
                total = 0.0
                for i in range(len(points) - 1):
                    lng1, lat1 = points[i]
                    lng2, lat2 = points[i + 1]
                    total += _haversine_km(lat1, lng1, lat2, lng2) * 1.3
                return round(total, 1)
        except Exception:
            pass
        # fallback: db.start_poi/end_poi 直线
        try:
            sp = self.db.start_poi or {}
            ep = self.db.end_poi or {}
            if sp.get("lat") and sp.get("lng") and ep.get("lat") and ep.get("lng"):
                return round(_haversine_km(sp["lat"], sp["lng"], ep["lat"], ep["lng"]) * 1.3, 1)
        except Exception:
            pass
        return 5.0

    def _find_driver(self, driver_id: str) -> dict:
        """Find driver by exact ID, index number, or name.
        Supports:
          - Exact ID: "D20001"
          - Index from last match_drivers result: "1", "2"
          - Driver name: "朱"
        """
        last_matched = getattr(self, "_last_matched_drivers", None) or []
        static_drivers = self.db.available_drivers or []

        # 1. Try exact driver_id match — prefer recently matched drivers
        for d in list(last_matched) + static_drivers:
            if d.get("driver_id") == driver_id:
                return d

        # 2. Try index-based match (from last match_drivers result)
        if last_matched:
            try:
                idx = int(driver_id)
                if 1 <= idx <= len(last_matched):
                    return last_matched[idx - 1]
            except (ValueError, TypeError):
                pass

            # 3. Try name-based match within last matched drivers
            for d in last_matched:
                if d.get("driver_name", "") == driver_id or driver_id in d.get("driver_name", ""):
                    return d

        # 4. Fallback: try name match in static drivers
        for d in static_drivers:
            if d.get("driver_name", "") == driver_id or driver_id in d.get("driver_name", ""):
                return d

        return None

    @is_tool(ToolType.WRITE)
    def create_ride_order(
        self,
        user_id: str,
        driver_id: str,
        start_address: str,
        end_address: str,
        via_addresses: Optional[List[str]] = None,
    ) -> str:
        """创建打车订单，支持途经点"""
        assert user_id, "User ID cannot be empty"
        assert self._check_user(user_id), "User ID does not match"
        assert driver_id, "Driver ID cannot be empty"
        assert start_address, "Start address cannot be empty"
        assert end_address, "End address cannot be empty"

        # Find the driver (supports ID, index, or name)
        driver_info = self._find_driver(driver_id)
        if driver_info is None:
            available_ids = [d.get("driver_id", "?") for d in (self.db.available_drivers or [])]
            return f"未找到司机 {driver_id}，当前可用司机ID: {', '.join(available_ids)}"

        order_id = self._generate_order_id(user_id)
        now = self.get_now("%Y-%m-%d %H:%M:%S")

        start_poi = self._effective_poi(self.db.start_poi, start_address)
        end_poi = self._effective_poi(self.db.end_poi, end_address)
        via_pois = [
            {"displayname": addr, "address": addr}
            for addr in (via_addresses or [])
            if addr
        ]

        order = RideOrder(
            order_id=order_id,
            user_id=user_id,
            driver_id=driver_info["driver_id"],
            driver_name=driver_info.get("driver_name", ""),
            plate_no=driver_info.get("plate_no", ""),
            car_type=driver_info.get("car_type", ""),
            brand_desc=driver_info.get("brand_desc", ""),
            car_model=driver_info.get("car_model", ""),
            start_poi=start_poi,
            end_poi=end_poi,
            estimate_fee=driver_info.get("estimate_fee", 0),
            discount_fee=driver_info.get("discount_fee", 0),
            recommend_score=driver_info.get("recommend_score", 0),
            tags=driver_info.get("tags", []),
            ota=driver_info.get("ota", 0),
            oea=driver_info.get("oea", 0),
            status="waiting_confirm",
            create_time=now,
            update_time=now,
            via_pois=via_pois,
        )

        if self.db.ride_orders is None:
            self.db.ride_orders = {}
        self.db.ride_orders[order_id] = order.to_dict()

        return repr(order)

    @is_tool(ToolType.WRITE)
    def confirm_ride(self, order_id: str) -> str:
        """确认用车"""
        assert order_id, "Order ID cannot be empty"
        assert self.db.ride_orders and order_id in self.db.ride_orders, f"Order {order_id} not found"

        order = self.db.ride_orders[order_id]
        if order["status"] == "confirmed":
            return f"订单 {order_id} 已经确认过了"
        if order["status"] == "cancelled":
            return f"订单 {order_id} 已取消，无法确认"

        order["status"] = "confirmed"
        order["update_time"] = self.get_now("%Y-%m-%d %H:%M:%S")
        self.db.ride_orders[order_id] = order

        ride_order = RideOrder(**order)
        return f"已确认用车。{repr(ride_order)}"

    @is_tool(ToolType.WRITE)
    def cancel_ride(self, order_id: str, reason: Optional[str] = "") -> str:
        """取消用车"""
        assert order_id, "Order ID cannot be empty"
        assert self.db.ride_orders and order_id in self.db.ride_orders, f"Order {order_id} not found"

        order = self.db.ride_orders[order_id]
        if order["status"] == "cancelled":
            return f"订单 {order_id} 已经取消过了"

        order["status"] = "cancelled"
        order["update_time"] = self.get_now("%Y-%m-%d %H:%M:%S")
        self.db.ride_orders[order_id] = order
        return f"订单 {order_id} 已取消。取消原因: {reason}" if reason else f"订单 {order_id} 已取消"

    @is_tool(ToolType.READ)
    def get_ride_order_detail(self, order_id: str) -> str:
        """获取打车订单详情"""
        assert order_id, "Order ID cannot be empty"
        assert self.db.ride_orders and order_id in self.db.ride_orders, f"Order {order_id} not found"
        order = self.db.ride_orders[order_id]
        ride_order = RideOrder(**order)
        return repr(ride_order)

    @is_tool(ToolType.READ)
    def get_ride_estimate(
        self,
        start_address: str,
        end_address: str,
        via_addresses: Optional[List[str]] = None,
    ) -> str:
        """获取打车预估价格信息，支持途经点（多目的地按累计距离计价）。

        返回每个车型的「预估价格 ¥X, 最高优惠 ¥Y」：
        - **预估价格 (estimate_fee)** 是**原价**（按车型基础费 + 累计距离单价计算），不含优惠。
        - **最高优惠 (discount_fee)** 是该车型最多可享的优惠金额。
        - **实付价格 = 预估价格 − 实际优惠**（实际优惠 ≤ 最高优惠）。

        若用户问"最便宜"且对优惠后价格敏感，agent 应推荐 (预估价格 − 最高优惠) 最低的车型。

        参数：
          - start_address / end_address：起点 / 终点地址
          - via_addresses：可选途经点列表（list[str]），按经过顺序传入；
            行程距离 = 起 → 途经1 → 途经2 → ... → 终 累计 km，价格随之变化。
            评估"哪种途经方案最便宜"时分别传不同 via_addresses 比较。

        同 (起点, 途经, 终点, 车型) 在多次调用、以及与 match_drivers / create_ride_order
        之间，预估价格保持一致（基于确定性单价）。
        """
        assert start_address, "Start address cannot be empty"
        assert end_address, "End address cannot be empty"

        # 与 match_drivers / get_trip_duration_estimate 一致：累计距离含途经点
        trip_distance_km = self._compute_trip_distance(start_address, end_address, via_addresses)

        car_types = self.db.available_car_types or ["快车", "专车", "出租车"]
        path_str = " → ".join([start_address] + list(via_addresses or []) + [end_address])
        head = f"行程距离: 约 {trip_distance_km} km"
        if via_addresses:
            head += f"（含 {len(via_addresses)} 个途经点）"
        results = [f"路径: {path_str}", head]
        for ct in car_types:
            fee = _estimate_fee(ct, trip_distance_km)
            max_discount = round(min(5.0, fee * 0.15), 1)
            results.append(
                f"{ct}: 预估价格 ¥{fee}, 最高优惠 ¥{max_discount}"
            )
        return "\n".join(results)

    @is_tool(ToolType.READ)
    def get_trip_duration_estimate(
        self,
        start_address: str,
        end_address: str,
        via_addresses: Optional[List[str]] = None,
    ) -> str:
        """估算从起点 → [途经点] → 终点的路程时间（分钟）和距离（公里）。

        与 match_drivers 不同：本工具用于反向时间推理（如反推预约用车的出发时刻）；
        match_drivers 只给当前可用司机的接驾 ETA，不适合反向时间推理。

        参数：
          - start_address / end_address：起点 / 终点地址（必填）
          - via_addresses：可选途经点列表，按顺序计算总距离 起→途经1→途经2→...→终

        路况说明：高峰判定**自动基于 system_time**（早 7-9 / 晚 17-19 点 → 1.5x 拥堵）。
        agent **无需也无法**通过参数控制高峰；同 task 任何调用同输入必同输出，跨 agent 100% 一致。
        agent 自行用工具返回的「距离 + 耗时」做反向推理（航班/会议时间 → 出发时刻）。
        """
        assert start_address, "Start address cannot be empty"
        assert end_address, "End address cannot be empty"

        # 计算行程距离（含途经点累计）
        trip_distance_km = self._compute_trip_distance(start_address, end_address, via_addresses)

        # 高峰判定：自动基于 task system_time（不再让 agent 传参影响）
        is_peak = False
        peak_window = ""
        sys_time = getattr(self.db, "time", None)
        if sys_time:
            try:
                hour = datetime.strptime(sys_time, "%Y-%m-%d %H:%M:%S").hour
                if 7 <= hour < 9:
                    is_peak = True
                    peak_window = "早高峰 (7:00-9:00)"
                elif 17 <= hour < 19:
                    is_peak = True
                    peak_window = "晚高峰 (17:00-19:00)"
            except (ValueError, TypeError):
                pass

        avg_speed_kmh = 17.0 if is_peak else 25.0
        est_min = max(5, round(trip_distance_km / avg_speed_kmh * 60))
        # 推荐缓冲：高峰 +10 分钟，平时 +5 分钟
        buffer_min = 10 if is_peak else 5

        # 路径 string：起 → 途经 → 终
        path_str = " → ".join([start_address] + list(via_addresses or []) + [end_address])
        lines = [
            f"路程预估：{path_str}",
            f"  距离: 约 {trip_distance_km} km" + (f"（含 {len(via_addresses)} 个途经点）" if via_addresses else ""),
            f"  预计耗时: 约 {est_min} 分钟",
        ]
        if is_peak:
            lines.append(f"  ⚠️ 当前 {peak_window}，已按拥堵时段估算")
        else:
            lines.append(f"  ✓ 当前非高峰时段")
        lines.append(f"  建议预留缓冲: +{buffer_min} 分钟（赶时间场景请加大裕量）")
        return "\n".join(lines)

    @is_tool(ToolType.WRITE)
    def create_scheduled_ride(
        self,
        user_id: str,
        driver_id: str,
        start_address: str,
        end_address: str,
        departure_time: str,
        via_addresses: Optional[List[str]] = None,
    ) -> str:
        """创建预约打车订单，支持途经点。参数 'departure_time'（YYYY-MM-DD HH:MM 或 'YYYY-MM-DD HH:MM:SS'）。"""
        assert user_id, "User ID cannot be empty"
        assert self._check_user(user_id), "User ID does not match"
        assert driver_id, "Driver ID cannot be empty"
        assert start_address, "Start address cannot be empty"
        assert end_address, "End address cannot be empty"
        assert departure_time, "Departure time cannot be empty"

        driver_info = self._find_driver(driver_id)
        if driver_info is None:
            available_ids = [d.get("driver_id", "?") for d in (self.db.available_drivers or [])]
            return f"未找到司机 {driver_id}，当前可用司机ID: {', '.join(available_ids)}"

        order_id = self._generate_order_id(user_id)
        now = self.get_now("%Y-%m-%d %H:%M:%S")

        start_poi = self._effective_poi(self.db.start_poi, start_address)
        end_poi = self._effective_poi(self.db.end_poi, end_address)
        via_pois = [
            {"displayname": addr, "address": addr}
            for addr in (via_addresses or [])
            if addr
        ]

        order = RideOrder(
            order_id=order_id,
            user_id=user_id,
            driver_id=driver_info["driver_id"],
            driver_name=driver_info.get("driver_name", ""),
            plate_no=driver_info.get("plate_no", ""),
            car_type=driver_info.get("car_type", ""),
            brand_desc=driver_info.get("brand_desc", ""),
            car_model=driver_info.get("car_model", ""),
            start_poi=start_poi,
            end_poi=end_poi,
            estimate_fee=driver_info.get("estimate_fee", 0),
            discount_fee=driver_info.get("discount_fee", 0),
            recommend_score=driver_info.get("recommend_score", 0),
            tags=driver_info.get("tags", []),
            ota=driver_info.get("ota", 0),
            oea=driver_info.get("oea", 0),
            status="scheduled",
            create_time=now,
            update_time=now,
            departure_time=departure_time,
            via_pois=via_pois,
        )

        if self.db.ride_orders is None:
            self.db.ride_orders = {}
        self.db.ride_orders[order_id] = order.to_dict()

        return repr(order)

    @is_tool(ToolType.WRITE)
    def create_ride_reminder(
        self,
        user_id: str,
        remind_time: str,
        start_address: str,
        end_address: str,
        via_addresses: Optional[List[str]] = None,
        note: Optional[str] = None,
    ) -> str:
        """占位 docstring（运行时由 tools_schema 覆盖）。"""
        assert user_id, "User ID cannot be empty"
        assert self._check_user(user_id), "User ID does not match"
        assert remind_time, "Remind time cannot be empty"
        assert start_address, "Start address cannot be empty"
        assert end_address, "End address cannot be empty"

        reminder_id = "RM" + get_hash(f"#REMINDER#{user_id}{datetime.now().timestamp()}")[:10]
        now = self.get_now("%Y-%m-%d %H:%M:%S")
        via_list = [a for a in (via_addresses or []) if a]

        reminder = {
            "reminder_id": reminder_id,
            "user_id": user_id,
            "remind_time": remind_time,
            "start_address": start_address,
            "end_address": end_address,
            "via_addresses": via_list,
            "note": note or "",
            "create_time": now,
            "status": "active",
        }
        if self.db.ride_reminders is None:
            self.db.ride_reminders = []
        self.db.ride_reminders.append(reminder)

        via_str = f"，途经 {' → '.join(via_list)}" if via_list else ""
        note_str = f"，备注：{note}" if note else ""
        return (
            f"已创建行程提醒 {reminder_id}：将于 {remind_time} 提醒您预约/叫车"
            f"「{start_address} → {end_address}{via_str}」{note_str}。"
        )

    @is_tool(ToolType.READ)
    def get_ride_real_time_status(self, order_id: str) -> str:
        """获取订单实时动态（司机位置、ETA、异常事件）"""
        assert order_id, "Order ID cannot be empty"
        assert self.db.ride_orders and order_id in self.db.ride_orders, f"Order {order_id} not found"

        order = self.db.ride_orders[order_id]
        if order["status"] not in ("confirmed", "scheduled"):
            return f"订单 {order_id} 状态为 {order['status']}，无实时动态"

        # Check for pre-configured events in db.ride_events
        event = "none"
        event_detail = "行程正常"
        driver_eta = 300
        driver_distance = 1500
        surge_multiplier = 1.0

        for ev in (self.db.ride_events or []):
            if ev.get("order_id") == order_id or ev.get("order_id") is None:
                event = ev.get("event", "none")
                event_detail = ev.get("event_detail", "")
                driver_eta = ev.get("driver_eta", driver_eta)
                driver_distance = ev.get("driver_distance", driver_distance)
                surge_multiplier = ev.get("surge_multiplier", surge_multiplier)
                break

        return json.dumps({
            "order_id": order_id,
            "driver_eta": driver_eta,
            "driver_distance": driver_distance,
            "surge_multiplier": surge_multiplier,
            "event": event,
            "event_detail": event_detail,
        }, ensure_ascii=False)

    @is_tool(ToolType.WRITE)
    def update_ride_destination(self, order_id: str, new_end_address: str) -> str:
        """修改打车订单目的地"""
        assert order_id, "Order ID cannot be empty"
        assert new_end_address, "New end address cannot be empty"
        assert self.db.ride_orders and order_id in self.db.ride_orders, f"Order {order_id} not found"

        order = self.db.ride_orders[order_id]
        if order["status"] == "cancelled":
            return f"订单 {order_id} 已取消，无法修改目的地"

        old_dest = order.get("end_poi", {}).get("displayname", "")
        order["end_poi"] = {"displayname": new_end_address, "address": new_end_address}
        order["update_time"] = self.get_now("%Y-%m-%d %H:%M:%S")
        self.db.ride_orders[order_id] = order

        ride_order = RideOrder(**order)
        return f"目的地已从「{old_dest}」更新为「{new_end_address}」。{repr(ride_order)}"

    @is_tool(ToolType.WRITE)
    def add_ride_note(self, order_id: str, note: str) -> str:
        """给打车订单添加备注信息（如特殊要求、乘客说明、物品说明等）"""
        assert order_id, "Order ID cannot be empty"
        assert note and note.strip(), "Note cannot be empty"
        assert self.db.ride_orders and order_id in self.db.ride_orders, f"Order {order_id} not found"

        order = self.db.ride_orders[order_id]
        existing_notes = order.get("notes", [])
        if isinstance(existing_notes, str):
            existing_notes = [existing_notes] if existing_notes else []
        existing_notes.append(note.strip())
        order["notes"] = existing_notes
        order["update_time"] = self.get_now("%Y-%m-%d %H:%M:%S")
        self.db.ride_orders[order_id] = order

        return f"备注已添加到订单 {order_id}：{note.strip()}。当前所有备注：{'; '.join(existing_notes)}"

    @is_tool(ToolType.READ)
    def get_history_orders(self, user_id: str) -> str:
        """获取用户历史打车订单（已完成的历史记录，可用于推荐车型）"""
        assert user_id, "User ID cannot be empty"
        assert self._check_user(user_id), "User ID does not match"

        if not self.db.history_orders:
            return "暂无历史打车订单"

        lines = [f"共{len(self.db.history_orders)}条历史订单："]
        for order in self.db.history_orders:
            lines.append(json.dumps(order, ensure_ascii=False))
        return "\n".join(lines)

    @is_tool(ToolType.READ)
    def get_flight_info(self, flight_no: str) -> str:
        """根据航班号查询航班起降信息（起飞城市/机场/时间 + 到达城市/机场/时间）。
        典型触发场景：用户提到航班号需要安排接机（起点=到达机场）或送机（终点=起飞机场）时，
        agent 应调本工具拿到具体机场名、到达时间，再决策叫车的起终点和出发时间。"""
        assert flight_no and flight_no.strip(), "Flight number cannot be empty"
        flight_no = flight_no.strip().upper()
        fi = self.db.flight_info
        if not fi:
            return f"未找到航班 {flight_no} 的信息（系统中无航班数据）"
        # 支持单个 dict 或 list
        if isinstance(fi, list):
            match = next((x for x in fi if (x.get('flight_no') or '').upper() == flight_no), None)
        elif isinstance(fi, dict):
            match = fi if (fi.get('flight_no') or '').upper() == flight_no else None
        else:
            match = None
        if not match:
            return f"未找到航班 {flight_no} 的信息"
        return (
            f"航班 {match['flight_no']}：\n"
            f"  起飞：{match.get('departure_time','?')}，{match.get('departure_city','?')}，{match.get('departure_airport','?')}\n"
            f"  到达：{match.get('arrival_time','?')}，{match.get('arrival_city','?')}，{match.get('arrival_airport','?')}"
        )

    @is_tool(ToolType.READ)
    def nearby_search(self, query: str, location: Optional[Dict[str, Any]] = None) -> str:
        """根据关键词在参考点附近搜索 POI 候选，返回名称/经纬度/距离（km）。

        参数：
          - query: 要搜索的内容（如 "咖啡馆"/"药店"/"三甲医院"/"火锅店" 等品类词）
          - location: 可选 dict 指定参考点，三选一：
              {"city_name": "海口"}  - 以该城市中心为参考
              {"place_name": "腾讯滨海大厦"}  - 以该地名为参考（先 address→coords 解析）
              {"lng": 110.32, "lat": 20.04}  - 直接给坐标
            未指定时默认以 task 当前位置（location[0]）为参考点
        返回：候选 POI 列表（name/lng/lat/distance_km），按距离升序，top 5
        """
        assert query and query.strip(), "Query cannot be empty"
        query = query.strip()

        # 1. 解析参考点
        ref_lng = ref_lat = None
        if location and isinstance(location, dict):
            if location.get("lng") is not None and location.get("lat") is not None:
                ref_lng, ref_lat = float(location["lng"]), float(location["lat"])
            elif location.get("place_name"):
                # 简单地名解析：从 task 环境中的 POI 匹配
                pn = location["place_name"]
                for poi in [self.db.start_poi, self.db.end_poi, self.db.home_address, self.db.company_address]:
                    if isinstance(poi, dict) and pn in (poi.get("displayname") or poi.get("address") or ""):
                        ref_lng, ref_lat = poi.get("lng"), poi.get("lat")
                        break
                if ref_lng is None:
                    for loc in (self.db.location or []):
                        addr = loc.get("address") if isinstance(loc, dict) else getattr(loc, "address", "")
                        if pn in addr:
                            ref_lng = loc.get("longitude") if isinstance(loc, dict) else getattr(loc, "longitude", None)
                            ref_lat = loc.get("latitude") if isinstance(loc, dict) else getattr(loc, "latitude", None)
                            break
        # 默认参考点 = task 当前位置
        if ref_lng is None and self.db.location:
            loc0 = self.db.location[0]
            ref_lng = loc0.get("longitude") if isinstance(loc0, dict) else getattr(loc0, "longitude", None)
            ref_lat = loc0.get("latitude") if isinstance(loc0, dict) else getattr(loc0, "latitude", None)
        if ref_lng is None:
            return f"无法确定搜索参考点（请通过 location 参数指定 lng/lat 或 place_name）"

        # 2. 从 task environment 的 nearby_pois 字段读候选（每 task 自带）
        nearby = getattr(self.db, "nearby_pois", None) or {}
        # 两层匹配（任一命中即纳入）：
        #   (a) category key 双向子串：query="儿童医院" 命中 key "医院"；query="咖啡" 命中 key "咖啡店"
        #   (b) POI name 子串：query="瑞幸咖啡" 命中 name "瑞幸咖啡(腾讯滨海大厦店)"
        # 按 (name, lng, lat) 去重，避免多 key/双层命中同一 POI 重复返回
        candidates = []
        seen_keys = set()
        for category, pois in nearby.items():
            cat_matches = (query == category or query in category or category in query)
            for p in pois:
                name = p.get("name") or ""
                name_matches = bool(name) and (query in name or (len(query) >= 2 and name in query))
                if not (cat_matches or name_matches):
                    continue
                key = (name, p.get("lng"), p.get("lat"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                candidates.append(p)
        if not candidates:
            return f"附近未找到与「{query}」匹配的 POI（task 环境未配置或类别不符）"

        # 3. 按距离排序
        for c in candidates:
            c_lng = c.get("lng")
            c_lat = c.get("lat")
            if c_lng is not None and c_lat is not None:
                c["_dist"] = _haversine_km(ref_lat, ref_lng, c_lat, c_lng)
            else:
                c["_dist"] = 9999
        candidates.sort(key=lambda x: x.get("_dist", 9999))
        top = candidates[:5]

        # 4. 持久化召回结果到 db.location，后续工具（get_trip_duration_estimate /
        # get_ride_estimate / create_ride_order 的 via_addresses 等）通过 _lookup_coords 可访问。
        # 与 search_pois 保持一致：不持久化，agent 用这些 POI 做 via_addresses 时会查不到坐标，
        # 整段行程退化为起→终直线，导致不同 via 路程相同（mt_04 类问题）。
        def _get_addr(loc):
            return loc.get('address') if isinstance(loc, dict) else getattr(loc, 'address', None)
        existing = {_get_addr(loc) for loc in (self.db.location or []) if _get_addr(loc)}
        if self.db.location is None:
            self.db.location = []
        for c in top:
            name = c.get('name')
            lng, lat = c.get('lng'), c.get('lat')
            if name and lng is not None and lat is not None and name not in existing:
                self.db.location.append({'address': name, 'longitude': lng, 'latitude': lat})
                existing.add(name)

        # 5. 格式化
        lines = [f"为「{query}」找到 {len(top)} 个附近 POI（按距离升序）："]
        for i, c in enumerate(top, 1):
            lines.append(
                f"[{i}] {c.get('name','?')} | 经纬度({c.get('lng','?')},{c.get('lat','?')}) | 距离 {c.get('_dist',0):.2f} km"
            )
        return "\n".join(lines)

    @is_tool(ToolType.READ)
    def search_along_route(
        self,
        query: str,
        start_address: str,
        end_address: str,
        max_detour_km: float = 2.0,
    ) -> str:
        """搜索从起点到终点路线上的 POI 候选（沿途搜）。

        参数：
          - query: 要搜索的内容（如「咖啡店」「加油站」「鲜花店」/ 品牌名「星巴克」/ 具体店名）
          - start_address: 起点地址
          - end_address: 终点地址
          - max_detour_km: 允许的最大绕路距离（km），默认 2.0；绕路 = (起→POI→终) − (起→终直线)

        与 nearby_search 的区别：
        - nearby_search：搜某一个**位置附近**的 POI（距单点距离升序）
        - search_along_route：搜**起→终路线上**的 POI（按绕路距离升序，反映"顺路程度"）

        典型场景：
        - 下班顺路买花 → search_along_route("鲜花店", 公司, 家)
        - 上班顺路买咖啡 → search_along_route("咖啡店", 家, 公司)

        返回：top 5 候选 POI（name / lng / lat / 绕路 km），并持久化到 db.location 供后续
        via_addresses 解析。
        """
        assert query and query.strip(), "Query cannot be empty"
        assert start_address, "start_address cannot be empty"
        assert end_address, "end_address cannot be empty"

        query = query.strip()

        # 1. 解析起终点坐标
        s_xy = self._lookup_coords(start_address)
        e_xy = self._lookup_coords(end_address)
        if s_xy is None or e_xy is None:
            return f"无法解析起点或终点坐标（请确认地址已通过 search_pois / nearby_search 召回）"
        direct_km = _haversine_km(s_xy[1], s_xy[0], e_xy[1], e_xy[0])

        # 2. 收集候选：与 nearby_search 同样的两层匹配（category key + POI name 子串）
        nearby = getattr(self.db, "nearby_pois", None) or {}
        candidates = []
        seen = set()
        for category, pois in nearby.items():
            cat_match = (query == category or query in category or category in query)
            for p in pois:
                name = p.get("name") or ""
                name_match = bool(name) and (query in name or (len(query) >= 2 and name in query))
                if not (cat_match or name_match):
                    continue
                key = (name, p.get("lng"), p.get("lat"))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(p)
        if not candidates:
            return f"沿路（{start_address} → {end_address}）未找到与「{query}」匹配的 POI（task 环境未配置或类别不符）"

        # 3. 算绕路距离
        for c in candidates:
            c_lng, c_lat = c.get("lng"), c.get("lat")
            if c_lng is None or c_lat is None:
                c["_detour"] = 9999
                continue
            d_via = (
                _haversine_km(s_xy[1], s_xy[0], c_lat, c_lng)
                + _haversine_km(c_lat, c_lng, e_xy[1], e_xy[0])
            )
            c["_detour"] = max(0.0, d_via - direct_km)
            c["_via_km"] = d_via

        # 4. 过滤 + 排序
        in_range = [c for c in candidates if c.get("_detour", 9999) <= max_detour_km]
        if not in_range:
            min_detour = min((c.get("_detour", 9999) for c in candidates), default=9999)
            return (
                f"沿路 {max_detour_km}km 绕路阈值内未找到与「{query}」匹配的 POI"
                f"（最接近的候选绕路 {min_detour:.2f}km，可放宽 max_detour_km 重试）"
            )
        in_range.sort(key=lambda x: x.get("_detour", 9999))
        top = in_range[:5]

        # 5. 持久化到 db.location（让后续 via_addresses 能解析）
        def _get_addr(loc):
            return loc.get('address') if isinstance(loc, dict) else getattr(loc, 'address', None)
        existing = {_get_addr(loc) for loc in (self.db.location or []) if _get_addr(loc)}
        if self.db.location is None:
            self.db.location = []
        for c in top:
            name = c.get('name')
            lng, lat = c.get('lng'), c.get('lat')
            if name and lng is not None and lat is not None and name not in existing:
                self.db.location.append({'address': name, 'longitude': lng, 'latitude': lat})
                existing.add(name)

        # 6. 格式化
        lines = [
            f"沿路（{start_address} → {end_address}，直线 {direct_km:.2f} km）"
            f"找到 {len(top)} 个匹配「{query}」的 POI（按绕路升序）："
        ]
        for i, c in enumerate(top, 1):
            lines.append(
                f"[{i}] {c.get('name','?')} | 经纬度({c.get('lng','?')},{c.get('lat','?')}) "
                f"| 绕路 +{c.get('_detour',0):.2f} km（经此点全程约 {c.get('_via_km',0):.2f} km）"
            )
        return "\n".join(lines)

    @is_tool(ToolType.READ)
    def get_current_location(self) -> str:
        """获取用户当前位置（地址、经纬度）。永远只返回单个结果（用户当前位置只有一个）。"""
        if not self.db.location:
            return "无法获取当前位置"

        loc = self.db.location[0]
        if isinstance(loc, dict):
            address, lng, lat = loc.get("address"), loc.get("longitude"), loc.get("latitude")
        else:
            address, lng, lat = loc.address, loc.longitude, loc.latitude
        return json.dumps(
            {"address": address, "longitude": lng, "latitude": lat},
            ensure_ascii=False,
        )

    @is_tool(ToolType.READ)
    def search_ride_orders(self, user_id: str, status: Optional[str] = None) -> str:
        """查询用户的打车订单"""
        assert user_id, "User ID cannot be empty"
        assert self._check_user(user_id), "User ID does not match"

        if not self.db.ride_orders:
            return "暂无打车订单"

        orders = []
        for oid, order in self.db.ride_orders.items():
            if order["user_id"] == user_id:
                if status is None or order["status"] == status:
                    ride_order = RideOrder(**order)
                    orders.append(repr(ride_order))

        if not orders:
            return "暂无符合条件的打车订单"
        return "\n".join(orders)
