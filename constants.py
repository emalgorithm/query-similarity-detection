from enum import Enum
from datetime import datetime


class Resolution(Enum):
    LOCATION_LEVEL_1 = "location_level_1"
    LOCATION_LEVEL_2 = "location_level_2"
    LOCATION_LEVEL_3 = "location_level_3"


class AlgorithmName(Enum):
    DENSITY = "density"
    MOBILITY = "mobility"


class Constants(object):
    LOCATION_LEVELS = [Resolution.LOCATION_LEVEL_1,
                       Resolution.LOCATION_LEVEL_2,
                       Resolution.LOCATION_LEVEL_2]

    LEVEL_1_LOCATIONS = ["Imola", "Faenza", "Castel Bolognese", "Lugo"]
    LEVEL_2_LOCATIONS = ["Bologna", "Ferrara", "Cesena", "Modena"]
    LEVEL_3_LOCATIONS = ["Emilia-Romagna", "Toscana", "Abruzzo"]

    MIN_DATE = "01/01/2016"
    MIN_TIMESTAMP = int(datetime.strptime(MIN_DATE, "%d/%m/%Y").timestamp())
    MAX_DATE = "31/12/2017"
    MAX_TIMESTAMP = int(datetime.strptime(MAX_DATE, "%d/%m/%Y").timestamp())

