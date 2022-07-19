import json
import shield

filename = 'learning_rate'
with open('cfgs/'+filename+'.json') as f:
   load_file = json.load(f)
for c_i in load_file:
   for cfg in load_file[c_i]:
      shield.main(cfg)

