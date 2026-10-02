"""Decal blacklists and HSRP national prefix definitions for filtering."""

NON_PLATE_WORDS: frozenset[str] = frozenset(
    {
        "GOOD", "GOODS", "LUCK", "CARRIER", "SPEED", "TATA", "ASHOK",
        "LEYLAND", "EICHER", "INDIAN", "NATIONAL", "PERMIT", "DIESEL",
        "STOP", "HORN", "PLEASE", "FAST", "SUPER", "INDIA", "IND",
        "ROAD", "LINES", "TRANSPORT", "MOTORS", "SUPREME", "CEMENT",
        "COACH", "AIR", "BRAKE", "ALL", "STATE", "40KM", "PUBLIC",
        "AUTO", "SAFETY", "FIRST", "FASTAG", "PASSING", "CAPACITY",
        "TARE", "GROSS", "OK", "WAIT", "SIDE", "JAI", "MATA", "SHREE",
        "CNG", "PETROL", "BHARATBENZ", "MAHINDRA", "VOLVO", "SCANIA",
        "POWER", "CABLE", "CABLES", "WIRE", "LOGISTICS", "ROADWAYS",
    }
)

COMMERCIAL_DECAL_SUBSTRINGS: tuple[str, ...] = (
    "CARRIER", "LEYLAND", "TRANSPORT", "NATIONALPERMIT", "FASTAG", "DIESEL",
)

HSRP_PREFIXES: tuple[str, ...] = ("IND", "1ND", "IN0", "IIND")
