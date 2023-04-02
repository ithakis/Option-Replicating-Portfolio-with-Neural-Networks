import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from math import ceil
import numpy as np
from time import perf_counter
import scipy.stats as stats
from scipy.stats import qmc
from tensorflow import keras
from tensorflow.keras import Model

import matplotlib.pyplot as plt
from matplotlib import ticker

from tqdm import trange

from numpy.random import seed   
from tensorflow.random import set_seed as tf_seed
tf_seed(1234) ; seed(1234)

def Replicating_Portfolio()