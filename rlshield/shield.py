import argparse
import copy

import numpy as np
import stormpy as sp
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import stormpy.pomdp
import random
import os.path
import inspect

# from rlshield.noshield import NoShield
# from rlshield.recorder import LoggingRecorder, VideoRecorder, StatsRecorder
# from rlshield.model_simulator import SimulationExecutor, Tracker
from noshield import NoShield
from recorder import LoggingRecorder, VideoRecorder, StatsRecorder
from model_simulator import Tracker
from model_simulator import SimulationExecutor
from rl_simulator import TF_Environment
import matplotlib.pyplot as plt



from gridstorm.plotter import Plotter
import gridstorm.models as models

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def compute_winning_region(model, formula, initial=True):
    options = sp.pomdp.IterativeQualitativeSearchOptions()
    model = sp.pomdp.prepare_pomdp_for_qualitative_search_Double(model, formula)
    solver = sp.pomdp.create_iterative_qualitative_search_solver_Double(model, formula, options)
    logger.info("compute winning region...")
    if initial:
        solver.compute_winning_policy_for_initial_states(100)
    else:
        solver.compute_winning_region(100)
    logger.info("done.")
    return solver.last_winning_region

def construct_otf_shield(model, winning_region):
    return sp.pomdp.BeliefSupportWinningRegionQueryInterfaceDouble(model, winning_region)

def build_pomdp(program, formula):
    options = stormpy.BuilderOptions([formula])
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    options.set_build_all_labels()
    options.set_build_all_reward_models()
    options.set_build_observation_valuations()
    logger.debug("Start building the POMDP")
    return sp.build_sparse_model_with_options(program, options)

experiment_to_grid_model_names = {
    "avoid": models.surveillance,
    "refuel": models.refuel,
    'obstacle': models.obstacle,
    "intercept": models.intercept,
    'evade': models.evade,
    'rocks': models.rocks
}

class ManualInput:
    def __init__(self, path, prop, constants):
        self.path = path
        self.properties = [prop]
        self.constants = constants

def main(cfg=None):
    if cfg:
        parser = argparse.ArgumentParser(description='The shielded POMDP simulator.')
        parser.add_argument('--prism', help="Specify model from prism file",default=False)
        parser.add_argument('--logfile', help="File to log to", default="rendering.log")
        parser.add_argument('--prop', help='Specify property string directly')
        parser.add_argument('--load-winning-region', '-wr', help="Load a winning region")
        parser.add_argument('--nr-finisher-runs', '-N', type=int, default=1)
        parser.add_argument('--video-path', help="Path for the video")
        parser.add_argument('--stats-path', help="Path for recording stats")
        parser.add_argument('--finishers-only', action='store_true')
        parser.add_argument('--seed', help="Seed for randomised movements", default=3)
        parser.add_argument('--title', help="Title for video")
        args = parser.parse_args()
        for c_i in cfg:
            if c_i == 'shield':
                args.__setattr__('noshield',cfg[c_i])
            else:
                args.__setattr__(c_i, cfg[c_i])
    else:
        parser = argparse.ArgumentParser(description='The shielded POMDP simulator.')
        model_group = parser.add_mutually_exclusive_group(required=True)
        model_group.add_argument('--grid-model', '-m', help=f'Model from the gridworld-by-storm visualisation set, choose from {str(experiment_to_grid_model_names.keys())}')
        model_group.add_argument('--prism', help="Specify model from prism file")
        parser.add_argument('--prop', help='Specify property string directly')
        parser.add_argument('--constants', '-c', help="Constants to select the instance of the model", default="")
        parser.add_argument('--load-winning-region', '-wr', help="Load a winning region")
        parser.add_argument('--maxsteps', '-s', help="Maximal number of steps", type=int, default=100)
        #parser.add_argument('--maxrendering', '-r', help='Maximal length of a rendering', type=int, default=100)
        parser.add_argument('--max-runs', '-NN', help="Number of runs", type=int, default=5000)
        parser.add_argument('--nr-finisher-runs', '-N', type=int, default=1)
        parser.add_argument('--video-path', help="Path for the video")
        parser.add_argument('--stats-path', help="Path for recording stats")
        parser.add_argument('--finishers-only', action='store_true')
        parser.add_argument('--seed', help="Seed for randomised movements", default=3)
        parser.add_argument('--title', help="Title for video")
        parser.add_argument('--logfile', help="File to log to", default="rendering.log")
        parser.add_argument('--noshield', help="Simulate without a shield", action='store_true')
        parser.add_argument('--obs_level',default='BELIEF_SUPPORT')
        parser.add_argument('--valuations',default=False)
        parser.add_argument('--learning_method',default='PPO')
        parser.add_argument('--eval-interval', type=int,default=100)
        parser.add_argument('--eval-episodes', type=int,default=5)
        parser.add_argument('--goal-value', type=int, default=10)

        args = parser.parse_args()
    logging.basicConfig(filename=f'{args.logfile}', filemode='w', level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    #logging.getLogger("rlshield.model_simulator").setLevel(logging.DEBUG)

    if args.prism and not args.prop:
        raise RuntimeError("Prism models require setting the property via --prop")
    if args.grid_model and args.prop:
        raise RuntimeError("Properties cannot be set manually when using the gridstorm models")

    if args.video_path is not None and not os.path.isdir(args.video_path):
        raise RuntimeError(f"Video path {args.video_path} not known!")
    if args.stats_path is not None and not os.path.isdir(args.stats_path):
        raise RuntimeError(f"Stats path {args.stats_path} not known!")

    if args.video_path and not args.grid_model:
        raise RuntimeError("Rendering is only supported for gridstorm models!")

    random.seed(args.seed)
    if args.grid_model:
        logger.info("Look up problem definition....")
        model = experiment_to_grid_model_names[args.grid_model]
        model_constants = list(inspect.signature(model).parameters.keys())
        if args.constants is None and len(model_constants) > 0:
            raise RuntimeError("Model constants {} defined, but not given by command line".format(",".join(model_constants)))
        constants = dict(item.split('=') for item in args.constants.split(","))
        input = model(**constants)
        # input.properties.append("Rmin=? [F \"goal\"]")
    else:
        input = ManualInput(args.prism, args.prop, args.constants)
        constants = dict(item.split('=') for item in args.constants.split(","))


    if args.load_winning_region:
        logger.info("Load winning region...")
        winning_region, preamble = stormpy.pomdp.BeliefSupportWinningRegion.load_from_file(args.load_winning_region)
        for line in preamble.split('\n'):
            if line == "":
                continue
            if line.startswith("model hash: "):
                hash = int(line[12:])
        compute_shield = False
    else:
        winning_region = None
        compute_shield = not args.noshield

    initial = False
    logger.info("Loading problem definition....")
    prism_program = sp.parse_prism_program(input.path)
    prop = sp.parse_properties_for_prism_program(input.properties[0], prism_program)[0]
    prism_program, props = stormpy.preprocess_symbolic_input(prism_program, [prop], input.constants)
    prop = props[0]
    prism_program = prism_program.as_prism_program()
    raw_formula = prop.raw_formula

    logger.info("Construct POMDP representation...")
    model = build_pomdp(prism_program, raw_formula)
    model = sp.pomdp.make_canonic(model)
    logger.info(model)
    #if model.hash() != hash:
    #    raise RuntimeError("Winning Region does not agree with Model")

    if compute_shield:
        winning_region = compute_winning_region(model, raw_formula, initial)

    if winning_region is not None:
        otf_shield = construct_otf_shield(model, winning_region)
    elif args.noshield:
        otf_shield = NoShield()
    else:
        logger.warning("No winning region: Shielding disabled.")
        otf_shield = NoShield()

    if args.load_winning_region:
        videoname = os.path.splitext(os.path.basename(args.load_winning_region))[0]
    else:
        constant_values = "-".join(constants.values())
        if compute_shield:
            videoname = f"{args.grid_model}-{constant_values}-computed-shield"
        else:
            videoname = f"{args.grid_model}-{constant_values}-noshield"

    tracker = Tracker(model, otf_shield)
    if args.video_path:
        renderer = Plotter(prism_program, input.annotations, model)
        if input.ego_icon is not None:
            renderer.load_ego_image(input.ego_icon.path, (0.6 / renderer._maxX))
        if args.title:
            renderer.set_title(args.title)
        recorder = VideoRecorder(renderer, only_keep_finishers=args.finishers_only)
        output_path = args.video_path
    elif args.stats_path:
        recorder = StatsRecorder(only_keep_finishers=args.finishers_only)
        output_path = args.stats_path
    else:
        logger.info("No video path set, rendering disabled.")
        output_path = None
        recorder = LoggingRecorder(only_keep_finishers=args.finishers_only)


    # executor = SimulationExecutor(model, tracker)
    # executor.simulate(recorder, total_nr_runs=1, nr_good_runs=1, maxsteps=args.maxsteps)
    # recorder.save(output_path, f"{videoname}")
    obs_type = args.obs_level
    valuations = args.valuations
    result_fname = f"_{obs_type}_valuations" if valuations else f"_{obs_type}"
    executor = TF_Environment(model, tracker,obs_length=1,maxsteps=args.maxsteps,obs_type=obs_type,valuations=valuations,goal_value=args.goal_value)
    eval_executor = TF_Environment(model,tracker,obs_length=1,maxsteps=args.maxsteps,obs_type=obs_type,valuations=valuations,goal_value=args.goal_value)
    print("Starting RL:\n")
    G0 = executor.simulate_deep_RL(recorder,total_nr_runs=args.max_runs, eval_interval=args.eval_interval,eval_episodes=args.eval_episodes, maxsteps=args.maxsteps,eval_env= eval_executor,agent_arg=args.learning_method)
    np.savetxt(f"{output_path}/{videoname}{result_fname}.csv",np.array(G0),delimiter=' ')
    recorder.save(output_path, f"{videoname}{result_fname}")

if __name__ == "__main__":
    main()
