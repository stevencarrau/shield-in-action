import json
import shield
import os

filename = 'DQN_long'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
with open('cfgs/'+filename+'.json') as f:
   load_file = json.load(f)
for c_i in load_file:
   for cfg in load_file[c_i]:
      shield.main(cfg)

