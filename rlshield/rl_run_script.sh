source ~/stormenv/bin/activate

declare -i MAX_STEPS=5000
declare -i EVAL_EPISODES=10
declare -i EVAL_INTERVAL=100
declare -a hold_ind=("0" "1")
declare -a Environments=("refuel" "refuel")
declare -a EpisodeLengths=(5 10 25 50 100 150 200 250)
declare -a Constants=("N=6,ENERGY=7" "N=6,ENERGY=8")
declare -a LearningMethods=("REINFORCE")


for i in "${!hold_ind[@]}"; do
echo ${Constants[$i]}
echo ${Environments[$i]}
for j in "${!EpisodeLengths[@]}"; do
echo ${EpisodeLengths[$j]}
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/REINFORCE -NN $MAX_STEPS --obs_level "BELIEF_SUPPORT" --learning_method "REINFORCE" --eval-interval $EVAL_INTERVAL  --eval-episodes $EVAL_EPISODES -s ${EpisodeLengths[$j]} 
	python shield.py --grid-model ${Environments[$i]} --constants ${Constants[$i]} --video-path newvideos/REINFORCE -NN $MAX_STEPS --obs_level "BELIEF_SUPPORT" --learning_method "REINFORCE" --eval-interval $EVAL_INTERVAL --eval-episodes $EVAL_EPISODES --noshield -s ${EpisodeLengths[$j]}
done
done
