from typing import List, Optional, Dict, Any

from pydantic import Field

from eff.data_model.tasks import Location
from eff.environment.db import DB


class Driver(object):
    """Represents a driver with vehicle information"""

    def __init__(
        self,
        driver_id: str,
        driver_name: str,
        plate_no: str,
        brand_desc: str,
        car_model: str,
        car_color: str,
        car_type: str,
        eta: int,
        distance: int,
        estimate_fee: float,
        discount_fee: float,
        recommend_score: int,
        tags: List[str],
        ota: float,
        oea: float,
    ):
        self.driver_id = driver_id
        self.driver_name = driver_name
        self.plate_no = plate_no
        self.brand_desc = brand_desc
        self.car_model = car_model
        self.car_color = car_color
        self.car_type = car_type
        self.eta = eta
        self.distance = distance
        self.estimate_fee = estimate_fee
        self.discount_fee = discount_fee
        self.recommend_score = recommend_score
        self.tags = tags
        self.ota = ota
        self.oea = oea

    def __repr__(self):
        return (
            f"Driver(driver_id={self.driver_id}, "
            f"driver_name={self.driver_name}, "
            f"plate_no={self.plate_no}, "
            f"brand_desc={self.brand_desc}, "
            f"car_model={self.car_model}, "
            f"car_color={self.car_color}, "
            f"car_type={self.car_type}, "
            f"eta={self.eta}秒, "
            f"distance={self.distance}米, "
            f"estimate_fee={self.estimate_fee}元, "
            f"discount_fee={self.discount_fee}元, "
            f"recommend_score={self.recommend_score}, "
            f"tags={self.tags}, "
            f"ota={self.ota}, "
            f"oea={self.oea})"
        )

    def to_dict(self):
        return {
            "driver_id": self.driver_id,
            "driver_name": self.driver_name,
            "plate_no": self.plate_no,
            "brand_desc": self.brand_desc,
            "car_model": self.car_model,
            "car_color": self.car_color,
            "car_type": self.car_type,
            "eta": self.eta,
            "distance": self.distance,
            "estimate_fee": self.estimate_fee,
            "discount_fee": self.discount_fee,
            "recommend_score": self.recommend_score,
            "tags": self.tags,
            "ota": self.ota,
            "oea": self.oea,
        }


class RideOrder(object):
    """Represents a ride-hailing order"""

    def __init__(
        self,
        order_id: str,
        user_id: str,
        driver_id: str,
        driver_name: str,
        plate_no: str,
        car_type: str,
        brand_desc: str,
        car_model: str,
        start_poi: dict,
        end_poi: dict,
        estimate_fee: float,
        discount_fee: float,
        recommend_score: int,
        tags: List[str],
        ota: float,
        oea: float,
        status: str,
        create_time: str,
        update_time: str,
        departure_time: Optional[str] = None,
        via_pois: Optional[List[dict]] = None,
        notes: Optional[List[str]] = None,
    ):
        self.order_id = order_id
        self.user_id = user_id
        self.driver_id = driver_id
        self.driver_name = driver_name
        self.plate_no = plate_no
        self.car_type = car_type
        self.brand_desc = brand_desc
        self.car_model = car_model
        self.start_poi = start_poi
        self.end_poi = end_poi
        self.estimate_fee = estimate_fee
        self.discount_fee = discount_fee
        self.recommend_score = recommend_score
        self.tags = tags
        self.ota = ota
        self.oea = oea
        self.status = status
        self.create_time = create_time
        self.update_time = update_time
        self.departure_time = departure_time
        self.via_pois = via_pois or []
        self.notes = notes or []

    def __repr__(self):
        departure = f", departure_time={self.departure_time}" if self.departure_time else ""
        via = ""
        if self.via_pois:
            names = [p.get("displayname", "") for p in self.via_pois]
            via = f", via_pois=[{', '.join(names)}]"
        return (
            f"RideOrder(order_id={self.order_id}, "
            f"user_id={self.user_id}, "
            f"driver_id={self.driver_id}, "
            f"driver_name={self.driver_name}, "
            f"plate_no={self.plate_no}, "
            f"car_type={self.car_type}, "
            f"brand_desc={self.brand_desc}, "
            f"car_model={self.car_model}, "
            f"start_poi={self.start_poi.get('displayname', '')}, "
            f"end_poi={self.end_poi.get('displayname', '')}"
            f"{via}, "
            f"estimate_fee={self.estimate_fee}元, "
            f"discount_fee={self.discount_fee}元, "
            f"recommend_score={self.recommend_score}, "
            f"tags={self.tags}, "
            f"ota={self.ota}, "
            f"oea={self.oea}, "
            f"status={self.status}, "
            f"create_time={self.create_time}, "
            f"update_time={self.update_time}"
            f"{departure}"
            f"{f', notes={self.notes}' if self.notes else ''})"
        )

    def to_dict(self):
        return {
            "order_id": self.order_id,
            "user_id": self.user_id,
            "driver_id": self.driver_id,
            "driver_name": self.driver_name,
            "plate_no": self.plate_no,
            "car_type": self.car_type,
            "brand_desc": self.brand_desc,
            "car_model": self.car_model,
            "start_poi": self.start_poi,
            "end_poi": self.end_poi,
            "estimate_fee": self.estimate_fee,
            "discount_fee": self.discount_fee,
            "recommend_score": self.recommend_score,
            "tags": self.tags,
            "ota": self.ota,
            "oea": self.oea,
            "status": self.status,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "departure_time": self.departure_time,
            "via_pois": self.via_pois,
            "notes": self.notes,
        }


class RidehailingDB(DB):
    """Database containing all ride-hailing related data"""

    home_address: Optional[Dict[str, Any]] = Field(
        default=None, description="User's home address POI info"
    )
    company_address: Optional[Dict[str, Any]] = Field(
        default=None, description="User's company address POI info"
    )
    available_drivers: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="List of available drivers"
    )
    ride_orders: Optional[Dict[str, Dict[str, Any]]] = Field(
        default={}, description="Dictionary of ride orders indexed by order_id"
    )
    available_car_types: Optional[List[str]] = Field(
        default=["快车", "专车", "出租车"], description="Available car types"
    )
    selected_car_types: Optional[List[str]] = Field(
        default=["快车", "专车", "出租车"], description="User selected car types"
    )
    start_poi: Optional[Dict[str, Any]] = Field(
        default=None, description="Start POI"
    )
    end_poi: Optional[Dict[str, Any]] = Field(
        default=None, description="End POI"
    )
    city_id: Optional[int] = Field(default=0, description="City ID")
    city_name: Optional[str] = Field(default="", description="City name")
    ride_events: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="Simulated ride events: driver_cancelled, eta_delay, surge_change"
    )
    history_orders: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="Historical completed ride orders (for user context)"
    )
    flight_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Flight info for get_flight_info tool. Schema: {flight_no, departure_time, departure_city, departure_airport, arrival_time, arrival_city, arrival_airport}"
    )
    ride_reminders: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="Ride reminders created at runtime by create_ride_reminder tool"
    )
    nearby_pois: Optional[Dict[str, Any]] = Field(
        default=None, description="Nearby POIs for location-based search, keyed by category"
    )

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        return {
            "num_drivers": len(self.available_drivers) if self.available_drivers else 0,
            "num_orders": len(self.ride_orders) if self.ride_orders else 0,
        }
