"""NFHS-5 State / Union Territory value labels (v024 / mv024).

Directly extracted from the NFHS-5 Stata definition files (IAIR7EFL.DO / IAMR7EFL.DO).
Code 26 is omitted in NFHS-5 (Dadra & Nagar Haveli merged with Daman & Diu as code 25).
"""
from typing import Dict, Optional

DEFAULT_STATE_CODE = 9
DEFAULT_STATE_NAME = "Uttar Pradesh"

NFHS5_STATE_OPTIONS: Dict[int, str] = {
    1: "Jammu & Kashmir",
    2: "Himachal Pradesh",
    3: "Punjab",
    4: "Chandigarh",
    5: "Uttarakhand",
    6: "Haryana",
    7: "NCT of Delhi",
    8: "Rajasthan",
    9: "Uttar Pradesh",
    10: "Bihar",
    11: "Sikkim",
    12: "Arunachal Pradesh",
    13: "Nagaland",
    14: "Manipur",
    15: "Mizoram",
    16: "Tripura",
    17: "Meghalaya",
    18: "Assam",
    19: "West Bengal",
    20: "Jharkhand",
    21: "Odisha",
    22: "Chhattisgarh",
    23: "Madhya Pradesh",
    24: "Gujarat",
    25: "Dadra & Nagar Haveli and Daman & Diu",
    27: "Maharashtra",
    28: "Andhra Pradesh",
    29: "Karnataka",
    30: "Goa",
    31: "Lakshadweep",
    32: "Kerala",
    33: "Tamil Nadu",
    34: "Puducherry",
    35: "Andaman & Nicobar Islands",
    36: "Telangana",
    37: "Ladakh",
}


def is_valid_state(code: Optional[int]) -> bool:
    return code in NFHS5_STATE_OPTIONS


def get_state_name(code: Optional[int]) -> Optional[str]:
    if code is None:
        return None
    return NFHS5_STATE_OPTIONS.get(code)
