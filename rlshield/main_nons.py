import json
import shield
import os

filenames = ['DQN','DDQN','REINFORCE','SAC','PPO']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for filename in filenames:
   with open('cfgs/'+filename+'.json') as f:
      load_file = json.load(f)
   for c_i in load_file:
      for cfg in load_file[c_i]:
         cfg['nonsparse'] = True
         cfg['video_path'] = 'newvideos/NonSparse/Partial/'+cfg['video_path'][10:]
         shield.main(cfg)

