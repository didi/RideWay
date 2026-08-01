import json
from typing import Optional

from eff.data_model.tasks import Task
from eff.domains.ridehailing.data_model import RidehailingDB
from eff.domains.ridehailing.tools import RidehailingTools
from eff.utils.utils import get_task_file_path
from eff.environment.environment import Environment


def get_environment(
    db: Optional[RidehailingDB] = None,
    language: str = None,
) -> Environment:
    if language is not None:
        from eff.utils.schema_utils import set_global_language
        set_global_language(language)

    if db is None:
        db = {}

    db_dict = {
        k: v for k, v in db.items()
        if k in [
            "time", "user_id", "weather", "location", "user_historical_behaviors",
            "orders", "home_address", "company_address", "available_drivers",
            "ride_orders", "available_car_types", "selected_car_types",
            "start_poi", "end_poi", "city_id", "city_name", "history_orders",
            "flight_info", "nearby_pois", "ride_reminders",
        ]
    }
    if "ride_orders" not in db_dict:
        db_dict["ride_orders"] = {}

    db = RidehailingDB.model_validate(db_dict)
    tools = RidehailingTools(db)
    from eff.environment.environment import get_agent_policy
    return Environment(
        domain_name="ridehailing",
        policy=get_agent_policy(language),
        tools=tools,
    )


# Default task sets: 44 session-derived tasks + 14 custom long-horizon tasks
DEFAULT_TASK_FILES = ["tasks_rh44_all.json", "tasks_custom14_all.json"]


def get_tasks(language: str = None) -> list[Task]:
    task_dir = get_task_file_path("ridehailing", language).parent
    tasks = []
    for filename in DEFAULT_TASK_FILES:
        with open(task_dir / filename, "r") as fp:
            tasks.extend(json.load(fp))
    return [Task.model_validate(task) for task in tasks]
