import os
import pandas as pd
import random
import itertools
import numpy as np

f_name = '/home/purushottam/Desktop/IIT_patna/FIRE/New/For_send/New_result_true.csv'
dt = pd.read_csv(f_name, header= 0, index_col = None)
df = dt.values
d1 = df[:,0]
d2 = df[:,1]

A = 0

for i in range(len(df)):
	if(d1[i] == d2[i]):
		A+=1

k = (A*100)/450
print(k)