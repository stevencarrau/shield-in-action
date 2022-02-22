source ~/storm_env/bin/activate
declare -i MAX_STEPS=10000
declare -i EVAL_EPISODES=100
declare -i EVAL_INTERVAL=1000
declare -a hold_ind=("0")
declare -a Environments=("intercept")
declare -a Constants=("N=7,RADIUS=1")
declare -a MaxRewards=("1000")
declare -a LearningMethods=("SAC")
declare -a ReplayLength=(0 2 5 10)


for i in "${!hold_ind[@]}"; do
echo ${Constants[$i]}
echo ${Environments[$i]}
for j in "${!LearningMethods[@]}"; do
for k in "${!ReplayLength[@]}"; do
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "OBS_LEVEL" --valuations "True" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES --goal-value ${MaxRewards[$j]} --replay-buffer-length ${ReplayLength[$k]}
#	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "BELIEF_SUPPORT" --valuations "False" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES --goal-value ${MaxRewards[$j]}
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "OBS_LEVEL" --valuations "True" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES --noshield --goal-value ${MaxRewards[$j]}
#	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "BELIEF_SUPPORT" --valuations "False" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES --noshield --goal-value ${MaxRewards[$j]}
done
done
done
