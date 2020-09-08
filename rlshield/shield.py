import argparse
import stormpy as sp
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import stormpy.pomdp
import random
import os.path
import inspect

from rlshield.noshield import NoShield
from rlshield.recorder import LoggingRecorder, VideoRecorder
from rlshield.model_simulator import SimulationExecutor, Tracker

from gridstorm.plotter import Plotter
import gridstorm.models as models

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def compute_winning_region(model, formula, initial=True):
    options = sp.pomdp.IterativeQualitativeSearchOptions()
    model = sp.pomdp.prepare_pomdp_for_qualitative_search_Double(model, formula)
    solver = sp.pomdp.create_iterative_qualitative_search_solver_Double(model, formula, options)
    logger.debug("compute winning region...")
    if initial:
        solver.compute_winning_policy_for_initial_states(100)
    else:
        solver.compute_winning_region(100)
    logger.debug("done.")
    return solver.last_winning_region

def construct_otf_shield(model, winning_region):
    return sp.pomdp.BeliefSupportWinningRegionQueryInterfaceDouble(model, winning_region)

def build_pomdp(program, formula):
    options = stormpy.BuilderOptions([formula])
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    options.set_build_all_labels()
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

def main():
    parser = argparse.ArgumentParser(description='Starter project for the shielded POMDP simulator.')
    parser.add_argument('--grid-model', '-m', help='Model from the gridworld-by-storm visualisation set', required=True)
    parser.add_argument('--constants', '-c', help="Constants to select the instance of the model")
    parser.add_argument('--load-winning-region', '-wr', help="Load a winning region")
    parser.add_argument('--maxsteps', '-s', help="Maximal number of steps", type=int, default=100)
    #parser.add_argument('--maxrendering', '-r', help='Maximal length of a rendering', type=int, default=100)
    parser.add_argument('--max-runs', '-NN', help="Number of runs", type=int, default=10000000)
    parser.add_argument('--nr-finisher-runs', '-N', type=int, default=1)
    parser.add_argument('--video-path', help="Path for the video")
    parser.add_argument('--finishers-only', action='store_true')
    parser.add_argument('--seed', help="Seed for randomised movements", default=3)
    parser.add_argument('--title', help="Title for video")
    parser.add_argument('--noshield', help="Simulate without a shield", action='store_true')

    logging.basicConfig(filename='rendering.log', filemode='w', level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    #logging.getLogger("rlshield.model_simulator").setLevel(logging.DEBUG)

    args = parser.parse_args()
    random.seed(args.seed)
    logger.info("Look up problem definition....")
    model = experiment_to_grid_model_names[args.grid_model]
    model_constants = list(inspect.signature(model).parameters.keys())
    if args.constants is None and len(model_constants) > 0:
        raise RuntimeError("Model constants {} defined, but not given by command line".format(",".join(model_constants)))
    constants = dict(item.split('=') for item in args.constants.split(","))
    input = model(**constants)

    if args.load_winning_region:
        logger.info("Load winning region...")
        winning_region, preamble = stormpy.pomdp.BeliefSupportWinningRegion.load_from_file(args.load_winning_region)
        for line in preamble.split('\n'):
            if line == "":
                continue
            if line.startswith("model hash: "):
                hash = int(line[12:])
    else:
        winning_region = None

    compute_shield = False
    initial = True
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
    print(model)
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
        if compute_shield:
            raise RuntimeWarning("Misleading names for videos.")
        constant_values = "-".join(constants.values())
        videoname = f"{args.grid_model}-{constant_values}-noshield"

    tracker = Tracker(model, otf_shield)
    if args.video_path:
        renderer = Plotter(prism_program, input.annotations, model)
        if input.ego_icon is not None:
            renderer.load_ego_image(input.ego_icon.path, (0.6 / renderer._maxX))
        if args.title:
            renderer.set_title(args.title)
        recorder = VideoRecorder(renderer, only_keep_finishers=args.finishers_only)
    else:
        logger.info("No video path set, rendering disabled.")
        recorder = LoggingRecorder(only_keep_finishers=args.finishers_only)

    executor = SimulationExecutor(model, tracker)
    executor.simulate(recorder, total_nr_runs=args.max_runs, nr_good_runs=args.nr_finisher_runs, maxsteps=args.maxsteps)
    recorder.save(args.video_path, f"{videoname}")

if __name__ == "__main__":
    main()
