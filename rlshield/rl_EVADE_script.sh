# source ~/storm_env/bin/activate
declare -i MAX_STEPS=5000
declare -i EVAL_EPISODES=100
declare -i EVAL_INTERVAL=100
declare -a hold_ind=("0" "1" "2" "3" "4" "5")
declare -a Environments=("evade")
declare -a Constants=("N=6,RADIUS=2")
declare -a MaxRewards=("10")
declare -a LearningMethods=("REINFORCE")


for i in "${!hold_ind[@]}"; do
echo ${Constants[$i]}
echo ${Environments[$i]}
for j in "${!LearningMethods[@]}"; do
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "OBS_LEVEL" --valuations "True" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES --goal-value ${MaxRewards[$j]} -s 450
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "BELIEF_SUPPORT" --valuations "False" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES --goal-value ${MaxRewards[$j]} -s 450
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "OBS_LEVEL" --valuations "True" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES --noshield --goal-value ${MaxRewards[$j]} -s 450
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/${LearningMethods[$j]} -NN $MAX_STEPS --obs_level "BELIEF_SUPPORT" --valuations "False" --learning_method ${LearningMethods[$j]} --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES --noshield --goal-value $ {MaxRewards[$j]} -s 450
done
done
