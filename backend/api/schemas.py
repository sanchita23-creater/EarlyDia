"""Request / response models for the EarlyDia API."""
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class ScreeningRequest(BaseModel):
    """Plain, user-level answers. Field values are validated in service.py (friendly messages);
    this model only enforces basic types. No raw survey variable names are required except the
    opaque option ids returned by GET /api/config."""
    model_config = ConfigDict(extra="ignore")

    sex: str
    age: int
    education: int
    residence: int
    tobacco_use: bool
    tobacco_products: List[str] = []
    cigarette_frequency: Optional[int] = None
    drinks_alcohol: bool
    alcohol_frequency: Optional[int] = None
    alcohol_years: Optional[float] = None
    alcohol_types: List[str] = []
    diet: Dict[str, int]
    height_cm: Optional[float] = None      # required for women (BMI); ignored for men
    weight_kg: Optional[float] = None
    bp_systolic: float
    bp_diastolic: float
    state: Optional[int] = None            # optional contextual state code (1-37), imputed if omitted
    model_version: Optional[str] = None    # "v1" or "v2"; defaults to v2
    # OPTIONAL - used only by the independent IDRS comparator (backend/api/idrs.py), never by the
    # ML models. All three together or none.
    waist_cm: Optional[float] = None
    physical_activity: Optional[str] = None
    family_history: Optional[str] = None


class ProfileLine(BaseModel):
    label: str
    value: str


class ScreeningResponse(BaseModel):
    sex: str
    probability: float
    percent: float
    cutoff: float
    cutoff_percent: float
    flagged: bool
    status: str
    profile: List[ProfileLine]
    guidance: List[dict]
    technical: dict
    idrs: Optional[dict] = None     # present only when the optional IDRS answers were given
    state_provided: bool = True
    state_used: Optional[str] = None
    warning: Optional[str] = None
