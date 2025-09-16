import os, json, csv
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import maptlotlib.pyplot as plt

# TODO
from config import SolverOpts
from metrics import (
    get_d_frob, krylov_metrics, angle_metrics, modewise_metrics
)
from signals import prbs, is_PE
from estimators.dmdc import DMDC
from estimators.moesp import MOESP
from run_single import c2d

def ginibre(n, m, rng):
    A = rng.normal(size=(n,n)) / np.sqrt(n)
    B = rng.normal(size=(n,m)) / np.sqrt(n)
    return A, B

