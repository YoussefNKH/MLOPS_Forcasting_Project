from pydantic import BaseModel
from typing import Optional, List


class PredictionInput(BaseModel):
    """Input schema for sales prediction"""
    id: int  # int16
    item_id: int  # int16
    dept_id: int  # int8
    cat_id: int  # int8
    store_id: int  # int8
    state_id: int  # int8
    d: int  # int16
    wm_yr_wk: int  # int16
    weekday: int  # int8
    wday: int  # int8
    month: int  # int8
    year: int  # int16
    event_name_1: int  # int8
    event_type_1: int  # int8
    event_name_2: int  # int8
    event_type_2: int  # int8
    snap_CA: int  # int8
    snap_TX: int  # int8
    snap_WI: int  # int8
    sell_price: float  # float16
    revenue: float  # float32
    sold_lag_1: float  # float16
    sold_lag_2: float  # float16
    sold_lag_3: float  # float16
    sold_lag_6: float  # float16
    sold_lag_12: float  # float16
    sold_lag_24: float  # float16
    sold_lag_36: float  # float16
    iteam_sold_avg: float  # float16
    state_sold_avg: float  # float16
    store_sold_avg: float  # float16
    cat_sold_avg: float  # float16
    dept_sold_avg: float  # float16
    cat_dept_sold_avg: float  # float16
    store_item_sold_avg: float  # float16
    cat_item_sold_avg: float  # float16
    dept_item_sold_avg: float  # float16
    state_store_sold_avg: float  # float16
    state_store_cat_sold_avg: float  # float16
    store_cat_dept_sold_avg: float  # float16
    rolling_sold_mean: float  # float16
    expanding_sold_mean: float  # float16
    selling_trend: float  # float16