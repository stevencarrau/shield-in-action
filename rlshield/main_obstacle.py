import shield

cfg = dict([["grid_model", "obstacle"],["learning_method", "REINFORCE"],
      ["max_runs", 5000],
      ["maxsteps", 100],
       ["constants", "N=6"],
        ["video_path", "newvideos/"],
         ["obs_level", "OBS_LEVEL"],
          ["eval_interval", 100],
           ["eval_episodes", 10],
            ["goal_value", 1000],
             ["valuations" , True],
              ["noshield", True],
               ["fixed_policy", False],
                ["prob" , 0.0]
])

shield.record_path(cfg)
