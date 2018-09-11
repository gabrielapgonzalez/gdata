import numpy as np
import pandas as pd
import logging
import io
import os
import sys

os.chdir('/dados/home-credit')

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s;%(levelname)s;%(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger = logging.getLogger('model_hcg')


train = np.load(ptrain.py)
teste = np.load(ptest.py)




