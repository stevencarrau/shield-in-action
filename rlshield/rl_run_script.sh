source ~/stormenv/bin/activate

declare -i MAX_STEPS=20000
declare -i EVAL_EPISODES=5
declare -i EVAL_INTERVAL=100
declare -a hold_ind=("0" "1" "2" "3" "4" "5")
declare -a Environments=("avoid" "evade" "intercept" "obstacle" "refuel" "rocks")
declare -a Constants=("N=6,RADIUS=3" "N=6,RADIUS=2" "N=7,RADIUS=1" "N=6" "N=6,ENERGY=8" "N=6")
declare -a LearningMethods=("DQN" "DDQN" "REINFORCE" "SAC" "PPO")


for i in "${!hold_ind[@]}"; do
echo ${Constants[$i]}
echo ${Environments[$i]}
for j in "${!LearningMethods[@]}"; do
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "OBS_LEVEL" --valuations "False" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "BELIEF_SUPPORT" --valuations "False" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "OBS_LEVEL" --valuations "False" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES --noshield
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "BELIEF_SUPPORT" --valuations "False" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES --noshield
done
done
