# !pip install shapely

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from os import path
import pickle
from shapely import geometry
from shapely.geometry import Point
from shapely.ops import cascaded_union, unary_union
from itertools import combinations
