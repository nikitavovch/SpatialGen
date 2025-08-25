import numpy as np
import shapely
from shapely.geometry import Polygon, MultiLineString
import trimesh
import torch
from numpy import typing as npt
from collections import defaultdict
from typing import Dict, Any, Union, Sequence

from src.utils.typing import *



# default color for unknown objects
DEFAULT_UNKNOWN_SEM2COLOR = { "label": "other", "color": np.array([100, 85, 144])}

WALL_COLOR = [120, 120, 120]
FLOOR_COLOR = [80, 50, 50]
CEILING_COLOR = [120, 120, 80]

color2labels_dict = {(204, 5, 255): 'bed',
 (146, 111, 194): 'nightstand',
 (7, 255, 255): 'wardrobe, closet, press',
 (6, 51, 255): 'chest of drawers, chest, bureau, dresser',
 (11, 102, 255): 'sofa',
 (0, 255, 112): 'coffee table',
 (224, 5, 255): 'cabinet',
 (10, 0, 255): 'swivel chair',
 (10, 255, 71): 'desk',
 (122, 0, 255): 'crt screen',
 (41, 0, 255): 'traffic light',
 (0, 173, 255): 'screen door, screen',
 (120, 120, 120): 'wall',
 (255, 6, 51): 'painting, picture',
 (255, 51, 7): 'curtain',
 (255, 9, 92): 'rug',
 (220, 220, 220): 'mirror',
 (80, 50, 50): 'floor',
 (255, 8, 41): 'column, pillar',
 (255, 112, 0): 'buffet, counter, sideboard',
 (194, 255, 0): 'bench',
 (0, 214, 255): 'stool',
 (255, 7, 71): 'shelf',
 (0, 153, 255): 'hood, exhaust hood',
 (0, 71, 255): 'street lamp',
 (0, 31, 255): 'chandelier',
 (0, 41, 255): 'sconce',
 (0, 133, 255): 'shower',
 (0, 255, 133): 'toilet, can, commode, crapper, pot, potty, stool, throne',
 (0, 163, 255): 'sink',
 (102, 8, 255): 'tub',
 (230, 230, 230): 'window ',
 (20, 255, 0): 'refrigerator, icebox',
 (120, 120, 80): 'ceiling',
 (8, 255, 214): 'armchair',
 (214, 255, 0): 'dishwasher',
 (255, 224, 0): 'stairs',
 (0, 255, 41): 'kitchen island',
 (150, 5, 61): 'person',
 (204, 255, 4): 'plant',
 (255, 122, 8): 'base, pedestal, stand',
 (250, 10, 15): 'fireplace',
 (0, 255, 194): 'tv',
 (0, 255, 173): 'computer',
 (51, 255, 0): 'stove',
 (7, 255, 224): 'seat',
 (255, 194, 7): 'cushion',
 (255, 0, 31): 'plaything, toy',
 (255, 214, 0): 'radiator',
 (0, 245, 255): 'fan',
 (255, 5, 153): 'signboard, sign',
 (180, 120, 120): 'building',
 (102, 255, 0): 'clock',
 (0, 122, 255): 'bannister, banister, balustrade, balusters, handrail',
 (92, 255, 0): 'basket, handbasket',
 (0, 10, 255): 'dirt track',
 (173, 0, 255): 'trash can',
 (0, 143, 255): 'countertop',
 (255, 163, 0): 'book',
 (255, 184, 6): 'fence',
 (184, 255, 0): 'bulletin board',
 (100, 85, 144): 'other'}
semid2labels_dict = {
    "1": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "2": {
        "label": "nightstand",
        "color": [
            146,
            111,
            194
        ]
    },
    "3": {
        "label": "wardrobe, closet, press",
        "color": [
            7,
            255,
            255
        ]
    },
    "4": {
        "label": "chest of drawers, chest, bureau, dresser",
        "color": [
            6,
            51,
            255
        ]
    },
    "5": {
        "label": "sofa",
        "color": [
            11,
            102,
            255
        ]
    },
    "6": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "7": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "8": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "9": {
        "label": "swivel chair",
        "color": [
            10,
            0,
            255
        ]
    },
    "10": {
        "label": "desk",
        "color": [
            10,
            255,
            71
        ]
    },
    "12": {
        "label": "crt screen",
        "color": [
            122,
            0,
            255
        ]
    },
    "101": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "102": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "103": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "104": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "106": {
        "label": "traffic light",
        "color": [
            41,
            0,
            255
        ]
    },
    "108": {
        "label": "screen door, screen",
        "color": [
            0,
            173,
            255
        ]
    },
    "109": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "110": {
        "label": "sofa",
        "color": [
            11,
            102,
            255
        ]
    },
    "113": {
        "label": "painting, picture",
        "color": [
            255,
            6,
            51
        ]
    },
    "114": {
        "label": "painting, picture",
        "color": [
            255,
            6,
            51
        ]
    },
    "115": {
        "label": "curtain",
        "color": [
            255,
            51,
            7
        ]
    },
    "116": {
        "label": "rug",
        "color": [
            255,
            9,
            92
        ]
    },
    "117": {
        "label": "mirror",
        "color": [
            220,
            220,
            220
        ]
    },
    "118": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "119": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "120": {
        "label": "rug",
        "color": [
            255,
            9,
            92
        ]
    },
    "121": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "122": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "123": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "124": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "125": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "126": {
        "label": "column, pillar",
        "color": [
            255,
            8,
            41
        ]
    },
    "135": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "136": {
        "label": "wardrobe, closet, press",
        "color": [
            7,
            255,
            255
        ]
    },
    "137": {
        "label": "nightstand",
        "color": [
            146,
            111,
            194
        ]
    },
    "139": {
        "label": "chest of drawers, chest, bureau, dresser",
        "color": [
            6,
            51,
            255
        ]
    },
    "140": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "141": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "142": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "143": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "145": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "146": {
        "label": "chest of drawers, chest, bureau, dresser",
        "color": [
            6,
            51,
            255
        ]
    },
    "147": {
        "label": "desk",
        "color": [
            10,
            255,
            71
        ]
    },
    "149": {
        "label": "desk",
        "color": [
            10,
            255,
            71
        ]
    },
    "150": {
        "label": "buffet, counter, sideboard",
        "color": [
            255,
            112,
            0
        ]
    },
    "151": {
        "label": "swivel chair",
        "color": [
            10,
            0,
            255
        ]
    },
    "152": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "154": {
        "label": "bench",
        "color": [
            194,
            255,
            0
        ]
    },
    "155": {
        "label": "stool",
        "color": [
            0,
            214,
            255
        ]
    },
    "157": {
        "label": "wardrobe, closet, press",
        "color": [
            7,
            255,
            255
        ]
    },
    "159": {
        "label": "shelf",
        "color": [
            255,
            7,
            71
        ]
    },
    "164": {
        "label": "hood, exhaust hood",
        "color": [
            0,
            153,
            255
        ]
    },
    "167": {
        "label": "street lamp",
        "color": [
            0,
            71,
            255
        ]
    },
    "168": {
        "label": "street lamp",
        "color": [
            0,
            71,
            255
        ]
    },
    "169": {
        "label": "chandelier",
        "color": [
            0,
            31,
            255
        ]
    },
    "170": {
        "label": "traffic light",
        "color": [
            41,
            0,
            255
        ]
    },
    "171": {
        "label": "sconce",
        "color": [
            0,
            41,
            255
        ]
    },
    "172": {
        "label": "shower",
        "color": [
            0,
            133,
            255
        ]
    },
    "173": {
        "label": "toilet, can, commode, crapper, pot, potty, stool, throne",
        "color": [
            0,
            255,
            133
        ]
    },
    "174": {
        "label": "sink",
        "color": [
            0,
            163,
            255
        ]
    },
    "175": {
        "label": "shower",
        "color": [
            0,
            133,
            255
        ]
    },
    "176": {
        "label": "tub",
        "color": [
            102,
            8,
            255
        ]
    },
    "179": {
        "label": "swivel chair",
        "color": [
            10,
            0,
            255
        ]
    },
    "180": {
        "label": "screen door, screen",
        "color": [
            0,
            173,
            255
        ]
    },
    "181": {
        "label": "screen door, screen",
        "color": [
            0,
            173,
            255
        ]
    },
    "182": {
        "label": "screen door, screen",
        "color": [
            0,
            173,
            255
        ]
    },
    "183": {
        "label": "screen door, screen",
        "color": [
            0,
            173,
            255
        ]
    },
    "185": {
        "label": "window ",
        "color": [
            230,
            230,
            230
        ]
    },
    "186": {
        "label": "window ",
        "color": [
            230,
            230,
            230
        ]
    },
    "187": {
        "label": "window ",
        "color": [
            230,
            230,
            230
        ]
    },
    "190": {
        "label": "refrigerator, icebox",
        "color": [
            20,
            255,
            0
        ]
    },
    "191": {
        "label": "ceiling",
        "color": [
            120,
            120,
            80
        ]
    },
    "192": {
        "label": "stool",
        "color": [
            0,
            214,
            255
        ]
    },
    "198": {
        "label": "other",
        "color": [
            100,
            85,
            144
        ]
    },
    "199": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "200": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "201": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "202": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "203": {
        "label": "sofa",
        "color": [
            11,
            102,
            255
        ]
    },
    "204": {
        "label": "sofa",
        "color": [
            11,
            102,
            255
        ]
    },
    "205": {
        "label": "sofa",
        "color": [
            11,
            102,
            255
        ]
    },
    "206": {
        "label": "sofa",
        "color": [
            11,
            102,
            255
        ]
    },
    "209": {
        "label": "armchair",
        "color": [
            8,
            255,
            214
        ]
    },
    "211": {
        "label": "sofa",
        "color": [
            11,
            102,
            255
        ]
    },
    "213": {
        "label": "swivel chair",
        "color": [
            10,
            0,
            255
        ]
    },
    "218": {
        "label": "toilet, can, commode, crapper, pot, potty, stool, throne",
        "color": [
            0,
            255,
            133
        ]
    },
    "225": {
        "label": "refrigerator, icebox",
        "color": [
            20,
            255,
            0
        ]
    },
    "226": {
        "label": "dishwasher",
        "color": [
            214,
            255,
            0
        ]
    },
    "227": {
        "label": "traffic light",
        "color": [
            41,
            0,
            255
        ]
    },
    "228": {
        "label": "crt screen",
        "color": [
            122,
            0,
            255
        ]
    },
    "234": {
        "label": "stool",
        "color": [
            0,
            214,
            255
        ]
    },
    "235": {
        "label": "stairs",
        "color": [
            255,
            224,
            0
        ]
    },
    "236": {
        "label": "screen door, screen",
        "color": [
            0,
            173,
            255
        ]
    },
    "237": {
        "label": "window ",
        "color": [
            230,
            230,
            230
        ]
    },
    "238": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "239": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "240": {
        "label": "ceiling",
        "color": [
            120,
            120,
            80
        ]
    },
    "241": {
        "label": "kitchen island",
        "color": [
            0,
            255,
            41
        ]
    },
    "243": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "244": {
        "label": "shelf",
        "color": [
            255,
            7,
            71
        ]
    },
    "246": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "247": {
        "label": "swivel chair",
        "color": [
            10,
            0,
            255
        ]
    },
    "251": {
        "label": "person",
        "color": [
            150,
            5,
            61
        ]
    },
    "253": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "254": {
        "label": "plant",
        "color": [
            204,
            255,
            4
        ]
    },
    "255": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "257": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "259": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "260": {
        "label": "base, pedestal, stand",
        "color": [
            255,
            122,
            8
        ]
    },
    "261": {
        "label": "fireplace",
        "color": [
            250,
            10,
            15
        ]
    },
    "262": {
        "label": "ceiling",
        "color": [
            120,
            120,
            80
        ]
    },
    "263": {
        "label": "ceiling",
        "color": [
            120,
            120,
            80
        ]
    },
    "264": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "265": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "266": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "267": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "269": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "270": {
        "label": "desk",
        "color": [
            10,
            255,
            71
        ]
    },
    "271": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "272": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "273": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "274": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "275": {
        "label": "shelf",
        "color": [
            255,
            7,
            71
        ]
    },
    "278": {
        "label": "tv",
        "color": [
            0,
            255,
            194
        ]
    },
    "279": {
        "label": "computer",
        "color": [
            0,
            255,
            173
        ]
    },
    "280": {
        "label": "stove",
        "color": [
            51,
            255,
            0
        ]
    },
    "281": {
        "label": "stove",
        "color": [
            51,
            255,
            0
        ]
    },
    "287": {
        "label": "mirror",
        "color": [
            220,
            220,
            220
        ]
    },
    "288": {
        "label": "mirror",
        "color": [
            220,
            220,
            220
        ]
    },
    "289": {
        "label": "mirror",
        "color": [
            220,
            220,
            220
        ]
    },
    "295": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "296": {
        "label": "seat",
        "color": [
            7,
            255,
            224
        ]
    },
    "297": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "302": {
        "label": "cushion",
        "color": [
            255,
            194,
            7
        ]
    },
    "305": {
        "label": "screen door, screen",
        "color": [
            0,
            173,
            255
        ]
    },
    "306": {
        "label": "window ",
        "color": [
            230,
            230,
            230
        ]
    },
    "307": {
        "label": "screen door, screen",
        "color": [
            0,
            173,
            255
        ]
    },
    "309": {
        "label": "window ",
        "color": [
            230,
            230,
            230
        ]
    },
    "312": {
        "label": "ceiling",
        "color": [
            120,
            120,
            80
        ]
    },
    "313": {
        "label": "traffic light",
        "color": [
            41,
            0,
            255
        ]
    },
    "315": {
        "label": "traffic light",
        "color": [
            41,
            0,
            255
        ]
    },
    "317": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "318": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "319": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "320": {
        "label": "toilet, can, commode, crapper, pot, potty, stool, throne",
        "color": [
            0,
            255,
            133
        ]
    },
    "324": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "328": {
        "label": "plaything, toy",
        "color": [
            255,
            0,
            31
        ]
    },
    "335": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "336": {
        "label": "screen door, screen",
        "color": [
            0,
            173,
            255
        ]
    },
    "337": {
        "label": "screen door, screen",
        "color": [
            0,
            173,
            255
        ]
    },
    "339": {
        "label": "chest of drawers, chest, bureau, dresser",
        "color": [
            6,
            51,
            255
        ]
    },
    "353": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "356": {
        "label": "radiator",
        "color": [
            255,
            214,
            0
        ]
    },
    "358": {
        "label": "sink",
        "color": [
            0,
            163,
            255
        ]
    },
    "366": {
        "label": "fan",
        "color": [
            0,
            245,
            255
        ]
    },
    "377": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "378": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "380": {
        "label": "signboard, sign",
        "color": [
            255,
            5,
            153
        ]
    },
    "381": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "383": {
        "label": "desk",
        "color": [
            10,
            255,
            71
        ]
    },
    "384": {
        "label": "swivel chair",
        "color": [
            10,
            0,
            255
        ]
    },
    "385": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "387": {
        "label": "shelf",
        "color": [
            255,
            7,
            71
        ]
    },
    "388": {
        "label": "buffet, counter, sideboard",
        "color": [
            255,
            112,
            0
        ]
    },
    "389": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "390": {
        "label": "traffic light",
        "color": [
            41,
            0,
            255
        ]
    },
    "391": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "395": {
        "label": "plaything, toy",
        "color": [
            255,
            0,
            31
        ]
    },
    "397": {
        "label": "building",
        "color": [
            180,
            120,
            120
        ]
    },
    "399": {
        "label": "signboard, sign",
        "color": [
            255,
            5,
            153
        ]
    },
    "404": {
        "label": "buffet, counter, sideboard",
        "color": [
            255,
            112,
            0
        ]
    },
    "408": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "409": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "413": {
        "label": "clock",
        "color": [
            102,
            255,
            0
        ]
    },
    "414": {
        "label": "clock",
        "color": [
            102,
            255,
            0
        ]
    },
    "425": {
        "label": "bannister, banister, balustrade, balusters, handrail",
        "color": [
            0,
            122,
            255
        ]
    },
    "426": {
        "label": "bannister, banister, balustrade, balusters, handrail",
        "color": [
            0,
            122,
            255
        ]
    },
    "427": {
        "label": "bannister, banister, balustrade, balusters, handrail",
        "color": [
            0,
            122,
            255
        ]
    },
    "448": {
        "label": "bannister, banister, balustrade, balusters, handrail",
        "color": [
            0,
            122,
            255
        ]
    },
    "449": {
        "label": "bannister, banister, balustrade, balusters, handrail",
        "color": [
            0,
            122,
            255
        ]
    },
    "450": {
        "label": "bannister, banister, balustrade, balusters, handrail",
        "color": [
            0,
            122,
            255
        ]
    },
    "451": {
        "label": "bannister, banister, balustrade, balusters, handrail",
        "color": [
            0,
            122,
            255
        ]
    },
    "457": {
        "label": "stove",
        "color": [
            51,
            255,
            0
        ]
    },
    "458": {
        "label": "refrigerator, icebox",
        "color": [
            20,
            255,
            0
        ]
    },
    "461": {
        "label": "dishwasher",
        "color": [
            214,
            255,
            0
        ]
    },
    "462": {
        "label": "sink",
        "color": [
            0,
            163,
            255
        ]
    },
    "464": {
        "label": "basket, handbasket",
        "color": [
            92,
            255,
            0
        ]
    },
    "466": {
        "label": "dirt track",
        "color": [
            0,
            10,
            255
        ]
    },
    "467": {
        "label": "shelf",
        "color": [
            255,
            7,
            71
        ]
    },
    "468": {
        "label": "trash can",
        "color": [
            173,
            0,
            255
        ]
    },
    "470": {
        "label": "refrigerator, icebox",
        "color": [
            20,
            255,
            0
        ]
    },
    "474": {
        "label": "mirror",
        "color": [
            220,
            220,
            220
        ]
    },
    "475": {
        "label": "basket, handbasket",
        "color": [
            92,
            255,
            0
        ]
    },
    "476": {
        "label": "dirt track",
        "color": [
            0,
            10,
            255
        ]
    },
    "477": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "478": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "479": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "480": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "481": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "482": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "483": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "484": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "485": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "486": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "487": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "488": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "489": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "490": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "491": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "492": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "498": {
        "label": "screen door, screen",
        "color": [
            0,
            173,
            255
        ]
    },
    "500": {
        "label": "chest of drawers, chest, bureau, dresser",
        "color": [
            6,
            51,
            255
        ]
    },
    "502": {
        "label": "countertop",
        "color": [
            0,
            143,
            255
        ]
    },
    "506": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "640": {
        "label": "stove",
        "color": [
            51,
            255,
            0
        ]
    },
    "641": {
        "label": "sink",
        "color": [
            0,
            163,
            255
        ]
    },
    "642": {
        "label": "dishwasher",
        "color": [
            214,
            255,
            0
        ]
    },
    "717": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "719": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "720": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "721": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "722": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "725": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "726": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "727": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "728": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "730": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "731": {
        "label": "countertop",
        "color": [
            0,
            143,
            255
        ]
    },
    "734": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "735": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "736": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "737": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "739": {
        "label": "shelf",
        "color": [
            255,
            7,
            71
        ]
    },
    "740": {
        "label": "countertop",
        "color": [
            0,
            143,
            255
        ]
    },
    "743": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "744": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "746": {
        "label": "bannister, banister, balustrade, balusters, handrail",
        "color": [
            0,
            122,
            255
        ]
    },
    "747": {
        "label": "bannister, banister, balustrade, balusters, handrail",
        "color": [
            0,
            122,
            255
        ]
    },
    "748": {
        "label": "bannister, banister, balustrade, balusters, handrail",
        "color": [
            0,
            122,
            255
        ]
    },
    "749": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "750": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "751": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "752": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "757": {
        "label": "dirt track",
        "color": [
            0,
            10,
            255
        ]
    },
    "758": {
        "label": "shelf",
        "color": [
            255,
            7,
            71
        ]
    },
    "761": {
        "label": "refrigerator, icebox",
        "color": [
            20,
            255,
            0
        ]
    },
    "762": {
        "label": "cushion",
        "color": [
            255,
            194,
            7
        ]
    },
    "763": {
        "label": "mirror",
        "color": [
            220,
            220,
            220
        ]
    },
    "764": {
        "label": "book",
        "color": [
            255,
            163,
            0
        ]
    },
    "766": {
        "label": "plaything, toy",
        "color": [
            255,
            0,
            31
        ]
    },
    "767": {
        "label": "plant",
        "color": [
            204,
            255,
            4
        ]
    },
    "769": {
        "label": "clock",
        "color": [
            102,
            255,
            0
        ]
    },
    "772": {
        "label": "painting, picture",
        "color": [
            255,
            6,
            51
        ]
    },
    "775": {
        "label": "cushion",
        "color": [
            255,
            194,
            7
        ]
    },
    "776": {
        "label": "cushion",
        "color": [
            255,
            194,
            7
        ]
    },
    "777": {
        "label": "rug",
        "color": [
            255,
            9,
            92
        ]
    },
    "877": {
        "label": "traffic light",
        "color": [
            41,
            0,
            255
        ]
    },
    "880": {
        "label": "traffic light",
        "color": [
            41,
            0,
            255
        ]
    },
    "889": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "890": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "892": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "893": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "894": {
        "label": "desk",
        "color": [
            10,
            255,
            71
        ]
    },
    "919": {
        "label": "trash can",
        "color": [
            173,
            0,
            255
        ]
    },
    "921": {
        "label": "dirt track",
        "color": [
            0,
            10,
            255
        ]
    },
    "923": {
        "label": "chest of drawers, chest, bureau, dresser",
        "color": [
            6,
            51,
            255
        ]
    },
    "924": {
        "label": "shelf",
        "color": [
            255,
            7,
            71
        ]
    },
    "925": {
        "label": "dirt track",
        "color": [
            0,
            10,
            255
        ]
    },
    "926": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "928": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "939": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "942": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "943": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "944": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "945": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "946": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "947": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "948": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "949": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "950": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "951": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "952": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "955": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "956": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "958": {
        "label": "countertop",
        "color": [
            0,
            143,
            255
        ]
    },
    "959": {
        "label": "countertop",
        "color": [
            0,
            143,
            255
        ]
    },
    "962": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "963": {
        "label": "sink",
        "color": [
            0,
            163,
            255
        ]
    },
    "975": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "976": {
        "label": "building",
        "color": [
            180,
            120,
            120
        ]
    },
    "977": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "978": {
        "label": "swivel chair",
        "color": [
            10,
            0,
            255
        ]
    },
    "979": {
        "label": "buffet, counter, sideboard",
        "color": [
            255,
            112,
            0
        ]
    },
    "980": {
        "label": "curtain",
        "color": [
            255,
            51,
            7
        ]
    },
    "981": {
        "label": "curtain",
        "color": [
            255,
            51,
            7
        ]
    },
    "982": {
        "label": "tub",
        "color": [
            102,
            8,
            255
        ]
    },
    "1123": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "1124": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "1126": {
        "label": "fence",
        "color": [
            255,
            184,
            6
        ]
    },
    "1127": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "1129": {
        "label": "building",
        "color": [
            180,
            120,
            120
        ]
    },
    "1131": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "1132": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "1133": {
        "label": "desk",
        "color": [
            10,
            255,
            71
        ]
    },
    "1134": {
        "label": "swivel chair",
        "color": [
            10,
            0,
            255
        ]
    },
    "1136": {
        "label": "building",
        "color": [
            180,
            120,
            120
        ]
    },
    "1137": {
        "label": "mirror",
        "color": [
            220,
            220,
            220
        ]
    },
    "1138": {
        "label": "buffet, counter, sideboard",
        "color": [
            255,
            112,
            0
        ]
    },
    "1139": {
        "label": "traffic light",
        "color": [
            41,
            0,
            255
        ]
    },
    "1140": {
        "label": "swivel chair",
        "color": [
            10,
            0,
            255
        ]
    },
    "1141": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "1142": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "1143": {
        "label": "buffet, counter, sideboard",
        "color": [
            255,
            112,
            0
        ]
    },
    "1146": {
        "label": "building",
        "color": [
            180,
            120,
            120
        ]
    },
    "1148": {
        "label": "plaything, toy",
        "color": [
            255,
            0,
            31
        ]
    },
    "1149": {
        "label": "building",
        "color": [
            180,
            120,
            120
        ]
    },
    "1153": {
        "label": "tub",
        "color": [
            102,
            8,
            255
        ]
    },
    "1154": {
        "label": "building",
        "color": [
            180,
            120,
            120
        ]
    },
    "1155": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "1156": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "1157": {
        "label": "cabinet",
        "color": [
            224,
            5,
            255
        ]
    },
    "1159": {
        "label": "stool",
        "color": [
            0,
            214,
            255
        ]
    },
    "2010": {
        "label": "sink",
        "color": [
            0,
            163,
            255
        ]
    },
    "2211": {
        "label": "wall",
        "color": [
            120,
            120,
            120
        ]
    },
    "2215": {
        "label": "window ",
        "color": [
            230,
            230,
            230
        ]
    },
    "2217": {
        "label": "crt screen",
        "color": [
            122,
            0,
            255
        ]
    },
    "2218": {
        "label": "crt screen",
        "color": [
            122,
            0,
            255
        ]
    },
    "2219": {
        "label": "crt screen",
        "color": [
            122,
            0,
            255
        ]
    },
    "2220": {
        "label": "crt screen",
        "color": [
            122,
            0,
            255
        ]
    },
    "2223": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "2224": {
        "label": "coffee table",
        "color": [
            0,
            255,
            112
        ]
    },
    "3008": {
        "label": "chandelier",
        "color": [
            0,
            31,
            255
        ]
    },
    "3009": {
        "label": "bed",
        "color": [
            204,
            5,
            255
        ]
    },
    "3011": {
        "label": "building",
        "color": [
            180,
            120,
            120
        ]
    },
    "3012": {
        "label": "bulletin board",
        "color": [
            184,
            255,
            0
        ]
    },
    "3013": {
        "label": "bulletin board",
        "color": [
            184,
            255,
            0
        ]
    },
    "3019": {
        "label": "building",
        "color": [
            180,
            120,
            120
        ]
    },
    "3020": {
        "label": "building",
        "color": [
            180,
            120,
            120
        ]
    },
    "3026": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3027": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3029": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3030": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3031": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3032": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3033": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3034": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3035": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3036": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3037": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3038": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3039": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3040": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3041": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3046": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3047": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3048": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3049": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3050": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3052": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3053": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3054": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3060": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3061": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3065": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3066": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3067": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3069": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3070": {
        "label": "floor",
        "color": [
            80,
            50,
            50
        ]
    },
    "3075": {
        "label": "sink",
        "color": [
            0,
            163,
            255
        ]
    },
    "3078": {
        "label": "shower",
        "color": [
            0,
            133,
            255
        ]
    },
    "65535": {
        "label": "other",
        "color": [
            100,
            85,
            144 
        ]
    }
}

# parse floor and veil corners
def parse_corners(corners: List[Dict[str, float]]) -> np.ndarray:
    corner_lst = []
    for corner in corners:
        corner_lst.append(
            [
                float(corner["start"]["x"]),
                float(corner["start"]["y"]),
                float(corner["start"]["z"]),
            ]
        )
        corner_lst.append(
            [
                float(corner["end"]["x"]),
                float(corner["end"]["y"]),
                float(corner["end"]["z"]),
            ]
        )
    return np.array(corner_lst)


def room_boundary_to_2d_polygon(boundary):
    room_region = to_polygon(boundary)
    room_region = room_region.simplify(tolerance=0.0001, preserve_topology=True)
    return room_region


def to_polygon(points):
    polygon = Polygon(points)
    # Check if the polygon is valid
    if not polygon.is_valid:
        # print("Invalid polygon:", explain_validity(polygon))
        # Attempt to fix the polygon
        polygon = polygon.buffer(0)
        if not polygon.is_valid:
            raise ValueError("Polygon could not be fixed and is still invalid.")
    return polygon


def room_meta_to_polygon(room_meta_dict: Dict[str, Any], SCALE: float = 0.001) -> Polygon:

    assert "floor" in room_meta_dict.keys() and "ceil" in room_meta_dict.keys()

    # calculate the roomlayout bbox

    floor_points_lst = parse_corners(room_meta_dict["floor"])
    ceil_points_lst = parse_corners(room_meta_dict["ceil"])
    assert len(floor_points_lst) == len(ceil_points_lst)
    assert len(floor_points_lst) % 2 == 0

    # find floor lines
    floor_lines = []
    floor_pts = []
    for i in range(len(floor_points_lst) // 2):
        floor_corner_i = floor_points_lst[i * 2]
        floor_corner_j = floor_points_lst[i * 2 + 1]
        floor_lines.append([floor_corner_i[:2] * SCALE, floor_corner_j[:2] * SCALE])
        floor_pts.append(floor_corner_i[:2] * SCALE)
        floor_pts.append(floor_corner_j[:2] * SCALE)
        # ceil_corner_i = ceil_points_lst[i * 2]
        # ceil_corner_j = ceil_points_lst[i * 2 + 1]

    try:
        # valid floor polygon
        floor_linestrs = shapely.line_merge(MultiLineString(floor_lines))
        floor_polygon = Polygon(floor_linestrs)
    except:
        floor_polygon = room_boundary_to_2d_polygon(floor_pts)
    return floor_polygon


def largest_rectangle_in_polygon(polygon, buffer=0.2):
    points = list(polygon.exterior.coords)
    # print(f'points : {len(points)}')
    xs = list(set([p[0] for p in points]))
    ys = list(set([p[1] for p in points]))

    max_area = 0
    best_rectangle = None

    for x1 in xs:
        for y1 in ys:
            for x2 in xs:
                if x2 == x1:
                    continue
                for y2 in ys:
                    if y2 == y1:
                        continue
                    rect = to_polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])
                    if rect.within(polygon):
                        area = rect.area
                        if area > max_area:
                            max_area = area
                            best_rectangle = rect

    points = list(best_rectangle.exterior.coords)
    x1, y1 = points[0]
    x2, y2 = points[2]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    wx = abs(x1 - x2)
    wy = abs(y1 - y2)
    buffer = min(min(wx, wy) / 6.0, buffer)
    wx -= buffer * 2
    wy -= buffer * 2
    x1 = cx - wx / 2
    x2 = cx + wx / 2
    y1 = cy - wy / 2
    y2 = cy + wy / 2
    buffer_rectangle = to_polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])
    return best_rectangle, buffer_rectangle, max_area


def shrink_or_swell_polygon(my_polygon, shrink_factor=0.10, swell=False):
    """returns the shapely polygon which is smaller or bigger by passed factor.
    If swell = True , then it returns bigger polygon, else smaller"""

    xs, ys = my_polygon.exterior.xy
    min_xs, min_ys = min(xs), min(ys)
    max_xs, max_ys = max(xs), max(ys)
    x_center = 0.5 * min_xs + 0.5 * max_xs
    y_center = 0.5 * min_ys + 0.5 * max_ys
    min_corner = shapely.geometry.Point(min_xs, min_ys)
    # max_corner = shapely.geometry.Point(max_xs, max_ys)
    center = shapely.geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * shrink_factor

    if swell:
        my_polygon_resized = my_polygon.buffer(shrink_distance)  # expand
    else:
        my_polygon_resized = my_polygon.buffer(-shrink_distance)  # shrink

    return my_polygon_resized

def convert_oriented_box_to_trimesh_fmt(box: Dict, 
                                        scale: float=1.0, 
                                        color_to_labels: Dict = None) -> trimesh.Trimesh:
    """ convert oriented box to mesh

    Args:
        box (Dict): 'id': , 'transform': , 'size': , 
        color_to_labels (Dict, optional): each category colors. Defaults to None.

    Returns:
        trimesh.Trimesh: mesh
    """
    box_sizes = np.array(box['size'])* scale
    transform_matrix = np.array(box["transform"]).reshape(4, 4)
    transform_matrix[:3, 3] = transform_matrix[:3, 3] * scale
    box_trimesh_fmt = trimesh.creation.box(box_sizes, transform_matrix)
    if color_to_labels is not None:
        labels_lst = list(color_to_labels.values())
        colors_lst = list(color_to_labels.keys())
        color = colors_lst[labels_lst.index(box['class'])]
    else:
        color = (np.random.random(3) * 255).astype(np.uint8).tolist()

    box_trimesh_fmt.visual.face_colors = color
    return box_trimesh_fmt

def create_oriented_bboxes(scene_bbox: List[Dict], scale: float=1.0) -> trimesh.Trimesh:
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box, scale=scale))

    mesh_list = trimesh.util.concatenate(scene.dump())
    return mesh_list

# 一些组合类别, 会造成很大bbox
COMPOSED_SEM_IDS = [202, 206, 269, 270, 316, 581, 666, 751, 752, 812, 814, 957, 2047, 2117, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3812]
BED_SEM_IDS = [1, 101, 199, 200, 201, 202, 257, 353, 725, 889, 939, 977, 1141, 3009 ]
CEIL_SEM_IDS = [191, 240, 262, 263, 312]
FLOOR_SEM_IDS = [118, 119, 238, 255, 506, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 962, 1156, 3026, 3027, 3029, 3030, 3031, 3032,3033,
3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3046, 3047, 3048, 3049, 3050, 3052, 3053, 3054, 3060, 3061, 3065, 3066, 3067, 3069, 3070]
WALL_SEM_IDS = [109, 121, 122, 123, 124, 239, 259, 264, 265, 266, 267, 295, 378, 389, 2211]
INVALID_OBJECT_SEM_IDS = CEIL_SEM_IDS + FLOOR_SEM_IDS + WALL_SEM_IDS + [65535]

def create_quadwall_to_trimesh_fmt(quad_vertices: np.array, 
                                normal: np.array = np.array([1,0,0]), 
                                color: np.array = np.array(WALL_COLOR),
                                camera_center: np.array = np.array([0, 0, 0]),
                                camera_rotation: np.array = np.eye(3)) ->trimesh.Trimesh:
    """
    create a quad polygen from vertices and normal
    params:
        quad_vertices: 4x3 np array, 4 vertices of the quad
        normal: 3x1 np array, normal of the quad
        camera_center: 3x1 np array, camera center
        camera_rotation: 3x3 np array, camera rotation
    """
    if camera_center is None:
        camera_center = np.array([0, 0, 0])
    if camera_rotation is None:
        camera_rotation = np.eye(3)
    quad_vertices = (quad_vertices - camera_center)
    quad_vertices = np.dot(camera_rotation, quad_vertices.T).T
    quad_triangles = []
    triangle = np.array([[0, 2, 1], [2, 0, 3]])
    quad_triangles.append(triangle)

    quad_triangles = np.concatenate(quad_triangles, axis=0)

    mesh = trimesh.Trimesh(vertices=quad_vertices,
                        faces=quad_triangles,
                        vertex_normals=np.tile(normal, (4, 1)),
                        face_colors=color,
                        process=False)
    return mesh

def parse_walls_from_meta(room_metadata: Dict[str, Any], room_folderpath: str = None) -> trimesh.Trimesh:
    # calculate the roomlayout bbox
    assert "floor" in room_metadata.keys() and "ceil" in room_metadata.keys()
    SCALE = 0.001
    # calculate the roomlayout bbox
    floor_points_lst = parse_corners(room_metadata["floor"])
    ceil_points_lst = parse_corners(room_metadata["ceil"])
    assert len(floor_points_lst) == len(ceil_points_lst)
    assert len(floor_points_lst) % 2 == 0

    # find floor lines
    floor_lines = []
    quad_wall_mesh_lst = []
    for i in range(len(floor_points_lst) // 2):
        floor_corner_i = floor_points_lst[i * 2]
        floor_corner_j = floor_points_lst[i * 2 + 1]
        floor_lines.append([floor_corner_i[:2] * SCALE, floor_corner_j[:2] * SCALE])

        ceil_corner_i = ceil_points_lst[i * 2]
        ceil_corner_j = ceil_points_lst[i * 2 + 1]
        # 3D coordinate for each wall
        quad_corners = np.array([floor_corner_i, ceil_corner_i, ceil_corner_j, floor_corner_j]) * SCALE
        wall_mesh = create_quadwall_to_trimesh_fmt(quad_corners)
        quad_wall_mesh_lst.append(wall_mesh)

    quad_wall_mesh_world = trimesh.util.concatenate(quad_wall_mesh_lst)
    return quad_wall_mesh_world
    
def parse_closed_room_from_meta(room_metadata: Dict[str, Any], room_folderpath: str = None, noceil_mesh: bool = False) -> trimesh.Trimesh:
    # calculate the roomlayout bbox
    assert "floor" in room_metadata.keys() and "ceil" in room_metadata.keys()
    SCALE = 0.001
    # calculate the roomlayout bbox
    floor_points_lst = parse_corners(room_metadata["floor"])
    ceil_points_lst = parse_corners(room_metadata["ceil"])
    assert len(floor_points_lst) == len(ceil_points_lst)
    assert len(floor_points_lst) % 2 == 0

    # find floor lines
    floor_lines, floor_corners = [], []
    quad_wall_mesh_lst = []
    avg_floor_h = 0.
    avg_ceil_h = 0.
    for i in range(len(floor_points_lst) // 2):
        floor_corner_i = floor_points_lst[i * 2] * SCALE
        floor_corner_j = floor_points_lst[i * 2 + 1] * SCALE
        floor_lines.append([np.around(floor_corner_i[:2], 3), np.around(floor_corner_j[:2], 3)])
        # floor_corners.append(np.around(floor_corner_i, 3))
        # floor_corners.append(np.around(floor_corner_j, 3))
        avg_floor_h += (floor_corner_i[2] + floor_corner_j[2]) / 2

        ceil_corner_i = ceil_points_lst[i * 2] * SCALE
        ceil_corner_j = ceil_points_lst[i * 2 + 1] * SCALE
        avg_ceil_h += (ceil_corner_i[2] + ceil_corner_j[2]) / 2
        
        # 3D coordinate for each wall
        quad_corners = np.array([floor_corner_i, ceil_corner_i, ceil_corner_j, floor_corner_j])
        wall_mesh = create_quadwall_to_trimesh_fmt(quad_corners, color=np.array(WALL_COLOR),)
        quad_wall_mesh_lst.append(wall_mesh)

    quad_wall_mesh_world = trimesh.util.concatenate(quad_wall_mesh_lst)

    ### complete floor and ceil
    # sort floor and ceiling lines by connection
    floor_linestrs = shapely.ops.linemerge(MultiLineString(floor_lines))
    # construct floor and ceiling polygons
    floor_polygon = Polygon(floor_linestrs)
    # convert polygon to mesh
    floor_vertices, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon)
    avg_floor_h = avg_floor_h / len(quad_wall_mesh_lst)
    floor_corners = [np.array([point[0], point[1], avg_floor_h]) for point in list(floor_polygon.exterior.coords)]
    floor_vertices_3d = np.concatenate([floor_vertices, avg_floor_h * np.ones((floor_vertices.shape[0], 1))], axis=1)
    floor_mesh = trimesh.Trimesh(vertices=floor_vertices_3d, faces=floor_faces, 
                                face_normals=np.array([[0, 0, 1]]).repeat(floor_faces.shape[0], axis=0),
                                face_colors=np.array([FLOOR_COLOR]).repeat(floor_faces.shape[0], axis=0))
    
    avg_ceil_h = avg_ceil_h / len(quad_wall_mesh_lst)
    ceiling_faces = floor_faces
    ceiling_vertices_3d = np.concatenate([floor_vertices, avg_ceil_h * np.ones((floor_vertices.shape[0], 1))], axis=1)
    ceiling_mesh = trimesh.Trimesh(vertices=ceiling_vertices_3d, faces=ceiling_faces, 
                                face_normals=np.array([[0, 0, -1]]).repeat(ceiling_faces.shape[0], axis=0),
                                face_colors=np.array([CEILING_COLOR]).repeat(ceiling_faces.shape[0], axis=0))
    # save scene layout as ply
    closed_room_mesh = trimesh.util.concatenate([floor_mesh, quad_wall_mesh_world] if noceil_mesh else [floor_mesh, quad_wall_mesh_world, ceiling_mesh])
    # room_mesh_filepath = os.path.join(room_folderpath, f"walls.ply")
    # closed_room_mesh.export(room_mesh_filepath)
    return closed_room_mesh, floor_corners, avg_ceil_h - avg_floor_h
        
def parse_obj_bbox_from_meta(room_metadata: Dict[str, Any], inst2sem_labels: Dict[str, Any] = None, room_folderpath: str = None, 
                             return_mesh=False, 
                             noceil_mesh=False) -> List:
    object_bbox_classes = []
    object_bbox_orientations = []
    object_bbox_translations = []
    object_bbox_extents = []
    object_is_composed = []    # 是否是组合家具
    
    obj_bbox_meshs = []
    cam0_dict = room_metadata["cameras"]["0"]
    if "bboxes" not in cam0_dict:
        return {}
    bbox_info_in_cam0 = cam0_dict["bboxes"]
    T_enu_cv = np.array(room_metadata["T_enu_cv"]).reshape(4, 4)
    T_cv_enu = np.linalg.inv(T_enu_cv)
    T_w2c0 = np.array(cam0_dict["camera_transform"]).reshape(4, 4)
    T_c02w = np.linalg.inv(T_w2c0)
    
    if room_folderpath is not None:
        save_bbox_mesh = True
    else:
        save_bbox_mesh = False
    for bbox_info in bbox_info_in_cam0:
        box_id = bbox_info["id"]
        class_id = inst2sem_labels.get(box_id, 65535)
        if class_id in INVALID_OBJECT_SEM_IDS:
            continue
        box_sizes = np.array(bbox_info['size'])
        # if np.any(box_sizes > 10) or np.any(box_sizes < 0.1):
        # skip bbox too large
        if np.all(box_sizes > 1.8):
            # print(f"skip bbox too large! box_sizes: {box_sizes}")
            continue
        # TODO: 这里需要处理组合家具
        if class_id in COMPOSED_SEM_IDS:
            # print(f"skip composed bbox! class_id: {class_id}")
            object_is_composed.append(class_id)
            # continue
        bbox_info["class"] = semid2labels_dict.get(str(class_id), DEFAULT_UNKNOWN_SEM2COLOR)['label']

        transform_matrix = np.array(bbox_info["transform"]).reshape(4, 4)
        # TODO: 这里需要确定该如何处理bbox很大的bed
        # if class_id in BED_SEM_IDS and box_sizes[2] > 1.2:
        #     print(f"contract on bed bbox! class_id: {class_id}")
        #     contract_box_sizes = np.array([box_sizes[0], box_sizes[1], box_sizes[2] * 0.5])
        #     bbox_center = transform_matrix[:3, 3]
        #     bbox_center[2] = bbox_center[2] - box_sizes[2] * 0.5
        #     transform_matrix[:3, 3] = bbox_center
        
        T_box2cam = T_cv_enu @ transform_matrix
        T_box2w = T_c02w @ T_box2cam
        bbox_info["transform"] = T_box2w.flatten().tolist()
        if save_bbox_mesh or return_mesh:
            bbox_world_mesh = convert_oriented_box_to_trimesh_fmt(bbox_info, color_to_labels=color2labels_dict)
            obj_bbox_meshs.append(bbox_world_mesh)
        
        bbox_object_orientations = np.array(T_box2w[:3, :3])
        bbox_object_translations = np.array(T_box2w[:3, 3])
        object_bbox_orientations.append(bbox_object_orientations)
        object_bbox_translations.append(bbox_object_translations)
        object_bbox_extents.append(box_sizes)
        object_bbox_classes.append(class_id)
    
    if  save_bbox_mesh or return_mesh:
        return_mesh_list = obj_bbox_meshs
        object_bbox_mesh = trimesh.util.concatenate(obj_bbox_meshs)
        # wall_mesh = trimesh.load_mesh(room_folderpath + "/room_wall.ply")
        wall_mesh, _, _ = parse_closed_room_from_meta(room_metadata, room_folderpath, noceil_mesh)
        object_bbox_mesh = trimesh.util.concatenate([object_bbox_mesh, wall_mesh])
        return_mesh_list.append(wall_mesh)
        if save_bbox_mesh:
            object_bbox_mesh.export(room_folderpath + "/layout_bbox.ply")
    object_bbox_classes = np.array(object_bbox_classes)
    object_bbox_orientations = np.array(object_bbox_orientations)
    object_bbox_translations = np.array(object_bbox_translations)
    object_bbox_extents = np.array(object_bbox_extents)
    object_is_composed = np.array(object_is_composed)
    
    object_bbox_dict = dict(obj_class=torch.from_numpy(object_bbox_classes).long(),
                            bbox_orientations=torch.from_numpy(object_bbox_orientations).float(),
                            bbox_positions=torch.from_numpy(object_bbox_translations).float(),
                            bbox_extents=torch.from_numpy(object_bbox_extents).float())
    obj_bbox = np.zeros((len(object_bbox_extents), 6))
    obj_bbox[:, 3:] = object_bbox_extents
    object_bbox_dict["obj_bbox"] = torch.from_numpy(obj_bbox).float()
    object_bbox_dict["boxes_oriented"] = True
    object_bbox_dict["composed_objects"] = torch.from_numpy(object_is_composed).long()
    if not return_mesh:
        return object_bbox_dict
    else:
        return object_bbox_dict, return_mesh_list
    
def parse_spatiallm_obj_bbox_from_meta(layout_meta_dict: Dict[str, Any], room_folderpath: str, return_mesh: bool = True):
        
    # visualize the layout
    object_bbox_classes = []
    object_bbox_labels = []
    object_bbox_orientations = []
    object_bbox_translations = []
    object_bbox_extents = []
    object_is_composed = []    # 是否是组合家具
    
    obj_bbox_meshs = []
    
    bbox_infos = layout_meta_dict['bboxes']
    for idx, bbox_info in enumerate(bbox_infos):
        box_sizes = np.array(bbox_info['size'])
        # skip bbox too large
        if np.all(box_sizes > 1.8):
            print(f"skip bbox too large! box_sizes: {box_sizes}")
            continue

        transform_matrix = np.array(bbox_info["transform"]).reshape(4, 4)
        T_box2w = transform_matrix
        bbox_world_mesh = convert_oriented_box_to_trimesh_fmt(bbox_info, color_to_labels=color2labels_dict)
        obj_bbox_meshs.append(bbox_world_mesh)
        
        bbox_object_orientations = np.array(T_box2w[:3, :3])
        bbox_object_translations = np.array(T_box2w[:3, 3])
        object_bbox_orientations.append(bbox_object_orientations)
        object_bbox_translations.append(bbox_object_translations)
        object_bbox_extents.append(box_sizes)
        object_bbox_classes.append(idx)
        object_bbox_labels.append(bbox_info['class'])
    
    return_mesh_list = obj_bbox_meshs
    object_bbox_mesh = trimesh.util.concatenate(obj_bbox_meshs)
    wall_mesh, floor_corners, wall_height = parse_closed_room_from_meta(layout_meta_dict, None)
    object_bbox_mesh = trimesh.util.concatenate([object_bbox_mesh, wall_mesh])
    return_mesh_list.append(wall_mesh)
    if return_mesh:
        object_bbox_mesh.export(room_folderpath + "/layout_bbox.ply")
    
    object_bbox_classes = np.array(object_bbox_classes)
    object_bbox_orientations = np.array(object_bbox_orientations)
    object_bbox_translations = np.array(object_bbox_translations)
    object_bbox_extents = np.array(object_bbox_extents)
    object_is_composed = np.array(object_is_composed)
    object_bbox_labels = np.array(object_bbox_labels)
    
    object_bbox_dict = dict(obj_class=torch.from_numpy(object_bbox_classes).long(),
                            bbox_orientations=torch.from_numpy(object_bbox_orientations).float(),
                            bbox_positions=torch.from_numpy(object_bbox_translations).float(),
                            bbox_extents=torch.from_numpy(object_bbox_extents).float())
    obj_bbox = np.zeros((len(object_bbox_extents), 6))
    obj_bbox[:, 3:] = object_bbox_extents
    object_bbox_dict["obj_bbox"] = torch.from_numpy(obj_bbox).float()
    object_bbox_dict["boxes_oriented"] = True
    object_bbox_dict["composed_objects"] = torch.from_numpy(object_is_composed).long()
    object_bbox_dict["obj_labels"] = object_bbox_labels
    object_bbox_dict["floor_corners"] = torch.from_numpy(np.array(floor_corners, dtype=np.float32)).float()
    object_bbox_dict["wall_height"] = torch.tensor(wall_height, dtype=torch.float32)
    if not return_mesh:
        return object_bbox_dict
    else:
        return object_bbox_dict, return_mesh_list
    
from pytorch3d.structures import Meshes

def trimesh_to_p3dmesh(trimesh_mesh: trimesh.Trimesh):
    vertices,faces,new_idx = trimesh.remesh.subdivide_to_size(trimesh_mesh.vertices, trimesh_mesh.faces, max_edge=0.2, return_index=True)
    vertices = torch.tensor(vertices,).float()
    faces = torch.tensor(faces,).long()
    face_color = (torch.tensor(trimesh_mesh.visual.face_colors[new_idx][...,:3])*1.0/255.0).view(1,trimesh_mesh.visual.face_colors[new_idx].shape[0],1,1,3)
    p3d_mesh = Meshes(verts=[vertices], faces=[faces], )
    
    return p3d_mesh,face_color


# sdc pytorch3d renderer for rendering segmentation, id?, depth, and scene coordinate map
from typing import Tuple
import torch
import torch.nn as nn

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.renderer import (
    RasterizationSettings, 
    MeshRasterizer
)

from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    Textures,
    TexturesUV,
    TexturesVertex,TexturesAtlas
)

class SDCMeshRenderer(nn.Module):
    def __init__(self, 
                 cameras: CamerasBase | None, 
                 image_size: Tuple[int],
                 blur_radius=1e-5,
                 max_faces_per_bin=None,
                 bin_size=None,
                 device="cuda") -> None:
        super().__init__()
        self.image_size = image_size
        raster_settings = RasterizationSettings(
            image_size=image_size, blur_radius=blur_radius, 
            faces_per_pixel=1, max_faces_per_bin=max_faces_per_bin,
            perspective_correct=True,
            bin_size=bin_size,
            cull_to_frustum=False,
            cull_backfaces =False,
            z_clip_value=-100,
            clip_barycentric_coords=False,
            )
        if cameras is None:
            R, T = look_at_view_transform(2.7, 0, 180) 
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    def to(self, device):
        self.rasterizer.to(device)
        return self

    def forward(self, 
                meshes, 
                faces_color: torch.Tensor|None = None,
                faces_id: torch.Tensor| None = None,
                cameras: CamerasBase | None = None,
                **kwargs) -> dict:
        """ Render meshes in a single scene as instance id.

        Args:
            meshes: mesh. 
        Returns:
            (H, W) int32
        """
        if cameras is None :
            R, T = look_at_view_transform(2.7, 0, 180) 
            cameras = FoVPerspectiveCameras(device=meshes.device, R=R, T=T)
        # self.rasterizer.cameras = cameras if cameras is not None else self.rasterizer.cameras

        fragments = self.rasterizer(meshes, cameras=cameras,**kwargs)

        pix_to_face_idx = fragments.pix_to_face 
        zbuf = fragments.zbuf
        
        rets = {}
        pix_to_face = fragments.pix_to_face.squeeze(-1)
        if faces_color is not None:
            render_seg = torch.zeros_like(pix_to_face).unsqueeze(-1).repeat(1,1,1,3).float()
            render_seg[pix_to_face!=-1] = faces_color.view(-1,3)[pix_to_face.flatten()[pix_to_face.flatten()!=-1]]
            rets["render_segment"] = render_seg
        if faces_id is not None:
            render_instance_id = torch.zeros_like(pix_to_face)
            render_instance_id[pix_to_face !=-1] = faces_id.flatten()[pix_to_face.flatten()[pix_to_face.flatten()!=-1]]
            rets["render_instance_id"] = render_instance_id
        if zbuf is not None:
            depth = zbuf
            rets["depth"] = depth
            # scene_coordinate_map =  cameras.unproject_points(depth,world_coordinates=True)
            # rets["scene_coordinate_map"] = scene_coordinate_map
        return rets


def project_3d_to_2d(points_3d, extrin, intrin):
    points_3d_homogeneous = torch.concat((points_3d, torch.ones([*points_3d.shape[:-1], 1]).to(points_3d)), -1)
    projection = intrin @ torch.linalg.inv(extrin)
    points_2d_homogeneous = points_3d_homogeneous @ projection.T
    points_2d = torch.concat([points_2d_homogeneous[..., :2] / points_2d_homogeneous[..., 2:3],
                              points_2d_homogeneous[..., 2:3]], axis=-1)
    return points_2d # [N, 8, 3]


def get_corner_bbox(bbox: torch.Tensor) -> torch.Tensor:
    min_corner = bbox[:, :3] - bbox[:, 3:] / 2
    max_corner = bbox[:, :3] + bbox[:, 3:] / 2
    dim_x, dim_y, dim_z = torch.split(max_corner - min_corner, 1, -1)
    dim_zeros = torch.zeros_like(dim_x)
    # get all 8 corners
    corners = torch.stack([
        min_corner,
        min_corner + torch.concat([dim_x, dim_zeros, dim_zeros], axis=-1),
        min_corner + torch.concat([dim_zeros, dim_y, dim_zeros], axis=-1),
        min_corner + torch.concat([dim_zeros, dim_zeros, dim_z], axis=-1),
        max_corner,
        max_corner - torch.concat([dim_x, dim_zeros, dim_zeros], axis=-1),
        max_corner - torch.concat([dim_zeros, dim_y, dim_zeros], axis=-1),
        max_corner - torch.concat([dim_zeros, dim_zeros, dim_z], axis=-1),
    ], axis=1)
    return corners

def filter_voxels_by_pixels(metas, **kwargs):
    voxels, voxel_size = metas['voxels'], metas['voxel_size']
    point_cloud_range = metas['point_cloud_range']
    # voxels: shape [M, 4], x y z label.
    voxels, labels = voxels[:, :3], voxels[:, -1]
    centroids_3d = voxels * voxel_size + voxel_size / 2 + point_cloud_range[:3]
    centroids_2d = project_3d_to_2d(centroids_3d, metas['extrin'], metas['intrin']) # [M, 3]
    valid = (centroids_2d[:, 0] > -(kwargs['width'] / 2)) & (centroids_2d[:, 0] < kwargs['width'] * 3 / 2 ) & \
            (centroids_2d[:, 1] > -(kwargs['height'] / 2)) & (centroids_2d[:, 1] < kwargs['height'] * 3 / 2) & \
            (centroids_2d[:, 2] > 0)
    return centroids_3d[valid], labels[valid]


def compute_camera_inside_bbox(pose, metas):
    oriented = metas.get('boxes_oriented', False)
    bbox, cls = metas['obj_bbox'], metas['obj_class']
    camera_xyz = pose[...,:3,3]

    if oriented:
        R_box2world = metas["bbox_orientations"]   # bbox_to_world [N, 3, 3]
        pos_box2world = metas["bbox_positions"]         # in world [N, 3]
        det = torch.det(R_box2world)
        if torch.any(det == 0):
            # Handle singular matrices
            print("Warning: Singular matrices detected in batch")
            epsilon = 1e-6 * torch.eye(R_box2world.size(-1), device=R_box2world.device)
            R_box2world = R_box2world + epsilon
        R_world_to_bbox = torch.linalg.inv(R_box2world)
        camera_xyz = (R_world_to_bbox[None, ...] @ (camera_xyz - pos_box2world[None, ...])[..., None])[..., 0]

    # compute ts
    bbox_min = bbox[:, :3] - bbox[:, 3:] / 2 # [N, 3]
    bbox_max = bbox[:, :3] + bbox[:, 3:] / 2

    origin_inside_bbox = (camera_xyz >= bbox_min) & (camera_xyz <= bbox_max) # [N, 3]
    origin_inside_bbox = origin_inside_bbox.all(-1) # [N]
    is_inside = origin_inside_bbox
    return is_inside

def compute_intersections_chunk(rays_o, rays_d, metas, return_depth: bool=False, voxelized: bool=False, **kwargs):
    oriented = metas.get('boxes_oriented', False)
    bbox, cls = metas['obj_bbox'], metas['obj_class']
    rays_o = rays_o[:, None]
    rays_d = rays_d[:, None]

    if oriented:
        R_box2world = metas["bbox_orientations"]   # bbox_to_world [N, 3, 3]
        pos_box2world = metas["bbox_positions"]         # in world [N, 3]
        R_world_to_bbox = torch.linalg.inv(R_box2world)
        # Rotate ray directions from world frame to the bbox frame
        rays_d = (R_world_to_bbox[None, ...] @ rays_d[..., None])[..., 0]
        rays_o = (R_world_to_bbox[None, ...] @ (rays_o - pos_box2world[None, ...])[..., None])[..., 0]

    # compute ts
    bbox_min = bbox[:, :3] - bbox[:, 3:] / 2 # [N, 3]
    bbox_max = bbox[:, :3] + bbox[:, 3:] / 2
    t_mins = (bbox_min - rays_o) / rays_d # [W*H, N, 3]
    t_maxs = (bbox_max - rays_o) / rays_d
    ts = torch.stack([t_mins, t_maxs], dim=-1) # [M, N, 3, 2]
    t_mins_max = ts.min(-1)[0].max(-1)[0] # [M, N] last of entrance
    t_maxs_min = ts.max(-1)[0].min(-1)[0] # [M, N] first of exit
    # The first of exit of intersected boxes should be in front of the image.
    is_intersects = (t_mins_max < t_maxs_min) & (t_maxs_min > 0) # [M, N]
    if not oriented and not metas.get("customed", False):
        corner_3d = get_corner_bbox(bbox) # [N, 8, 3]
        if oriented:
            # Transform points from bbox frame to world frame
            corner_3d = (R_box2world[:, None] @ corner_3d[..., None])[..., 0] + pos_box2world[:, None]
        corner_2d = project_3d_to_2d(corner_3d, metas['extrin'], metas['intrin'])
        # Filter the case the camera inside the box and the box behined the camera
        unseen_bbox = torch.all(corner_2d[..., 2] <= 0, dim=1) # [N]
        origin_inside_bbox = (rays_o[0] >= bbox_min) & (rays_o[0] <= bbox_max) # [N, 3]
        origin_inside_bbox = origin_inside_bbox.all(-1) # [N]
        is_intersects[:, unseen_bbox | origin_inside_bbox] = False
        del corner_2d, corner_3d, unseen_bbox, origin_inside_bbox
    del t_mins, t_maxs, t_mins_max, t_maxs_min

    # Only care about the rays that intersects boxes > 0
    keep = is_intersects.sum(-1) > 0 # [M]
    t_nears = ts.min(-1)[0][keep] # [L, N, 3]
    rays_o = rays_o[keep] # [L, 1, 3]
    rays_d = rays_d[keep] # [L, 1, 3]
    intersects = rays_o[..., None, :] + t_nears[..., None] * rays_d[..., None, :] # [L, N, 3, 3]
    eps = torch.tensor([1e-4, 1e-4, 1e-4], dtype=torch.float32, device=bbox_min.device)
    bbox_min_expanded = (bbox_min[..., None, :] - eps).repeat(1, 3, 1)
    bbox_max_expanded = (bbox_max[..., None, :] + eps).repeat(1, 3, 1)
    valid_intersects = (intersects >= bbox_min_expanded) & (intersects <= bbox_max_expanded)
    valid_intersects = valid_intersects.all(-1) # [L, N, 3]
    is_positive = t_nears >= 0 # [L, N, 3]
    valid_intersects &= is_positive
    del intersects, is_positive, bbox_min_expanded, bbox_max_expanded

    # Find the nearest valid intersected plane and nearest intersected bbox.
    t_nears[~valid_intersects] = 1e10
    del valid_intersects
    t_nears = t_nears.min(-1)[0] # [L, N]
    sorted_min = torch.argsort(t_nears, dim=-1) # [L, N]
    first_min, second_min = sorted_min[:, 0], sorted_min[:, 1] if sorted_min.shape[1] > 1 else None # [L]

    return_dict = dict()
    # assign class index to pixel
    nearest_bbox_idx = torch.zeros_like(is_intersects[:, 0]).long() - 1 # [M]
    nearest_bbox_idx[keep] = first_min.long()
    return_dict.update(nearest_bbox_idx=nearest_bbox_idx)

    # assign depth value to pixel
    if return_depth:
        nearest_distance_2d = torch.zeros_like(is_intersects[:, 0]).to(bbox_min) - 1 # [M]
        nearest_distance_2d[keep] = torch.gather(t_nears, dim=1, index=first_min.unsqueeze(-1)).squeeze(-1)
        return_dict.update(nearest_distance_2d=nearest_distance_2d)

    return return_dict

def compute_intersections(rays_o, rays_d, metas, return_depth: bool=True, wh: bool=False, **kwargs
                            ) -> Union[npt.NDArray, Sequence[npt.NDArray]]:
    """
    Compute the intersection of rays with the bounding boxes.
    get the nearest bbox index (semantic) and the distance to the camera (depth)
    """
    voxelized = False
    wh_shape = rays_o.shape
    if 'voxels' in metas:
        voxelized = True
        filtered_voxels, filtered_labels = filter_voxels_by_pixels(metas, **kwargs)
        filtered_voxels = torch.concat([filtered_voxels, torch.zeros_like(filtered_voxels) + metas['voxel_size']], dim=1)
        metas.update(obj_bbox=filtered_voxels, obj_class=filtered_labels)
        del filtered_voxels, filtered_labels

    rays_d[rays_d == 0] = 1e-8
    rays_o_flattened = rays_o.reshape(-1, 3) # [M, 3]
    rays_d_flattened = rays_d.reshape(-1, 3)
    del rays_o, rays_d

    rays_chunk = int(kwargs.get('rays_chunk', rays_o_flattened.shape[0]))
    all_outputs = defaultdict(list)
    for i in range(0, rays_o_flattened.shape[0], rays_chunk):
        chunk_outputs = compute_intersections_chunk(rays_o_flattened[i : i + rays_chunk],
                                                            rays_d_flattened[i : i + rays_chunk],
                                                            metas, return_depth, voxelized, **kwargs)
        for output_name, output in chunk_outputs.items():
            all_outputs[output_name].append(output)
    all_outputs = {k: torch.cat(v) for k, v in all_outputs.items()}

    nearest_bbox_idx_2d = all_outputs['nearest_bbox_idx'].reshape(wh_shape[:2]) # [W, H]
    depth_image = all_outputs['nearest_distance_2d'].reshape(wh_shape[:2]).cpu().numpy() if return_depth else None
    index_image = nearest_bbox_idx_2d.cpu().numpy()
    nearest_bbox_idx_2d[nearest_bbox_idx_2d != -1] = metas['obj_class'][nearest_bbox_idx_2d[nearest_bbox_idx_2d != -1]].long()
    nearest_bbox_idx_2d[nearest_bbox_idx_2d == -1] = 0
    label_image = nearest_bbox_idx_2d.cpu().numpy()
    del all_outputs, nearest_bbox_idx_2d

    if not wh:
        label_image = label_image.transpose(-1, -2)
        index_image = index_image.transpose(-1, -2)
        depth_image = depth_image.transpose(-1, -2) if depth_image is not None else None

    return label_image, index_image, depth_image
