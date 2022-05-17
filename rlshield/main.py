import json
import shield

filename = 'avoid'
with open('cfgs/'+filename+'.json') as f:
   load_file = json.load(f)
for c_i in load_file:
   for cfg in load_file[c_i]:
      shield.main(cfg)

for policy in ["qmdp", "mdp"]:
   cfg["policy"] = policy
   exp = Experiment(filename + "_" + policy, cfg, 10)
   exp.execute(False)
