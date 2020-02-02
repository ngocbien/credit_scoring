import os
print(os.getcwd())
from config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(path_to_file+'/train.csv')
print(df.shape)