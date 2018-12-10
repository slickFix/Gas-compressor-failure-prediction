
# Importing libraries

import numpy as np
import pandas as pd
import os

# changing to dataset location
os.getcwd()
os.chdir('/home/siddharth/Downloads/Dataset/P_projects/Compressor data')


# reading dataset
df  = pd.read_excel('30k_records.xlsx')