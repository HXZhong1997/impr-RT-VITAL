import pickle
import numpy as np
import os

model_name='impr-rt-vital'

save_path=os.path.join('./result',model_name)
if os.path.exists(save_path) == False:
    os.mkdir(save_path)

with open('./result/results.dict','rb') as f:
    result=pickle.load(f)

result_bb=result['bb_result']
result_nobb=result['bb_result_nobb']

if os.path.exists(save_path)==False:
   os.mkdir(save_path)

for (key,val) in result_bb.items():
    np.savetxt(os.path.join(save_path,key+'.txt'), val)
