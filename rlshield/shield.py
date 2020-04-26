import argparse
import stormpy as sp
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import stormpy.pomdp
import random

from rlshield.noshield import NoShield
from rlshield.recorder import LoggingRecorder, VideoRecorder
from rlshield.model_simulator import SimulationExecutor, Tracker

from gridstorm.plotter import Plotter
from gridstorm.annotations import ProgramAnnotation

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.INFO)


def compute_winning_region(model, formula):
    options = sp.pomdp.IterativeQualitativeSearchOptions()
    solver = sp.pomdp.create_iterative_qualitative_search_solver_Double(model, formula, options)
    ## TODO select a good lookahead
    solver.compute_winning_region(model.nr_states)
    return solver.last_winning_region

def construct_otf_shield(model, winning_region):
    return sp.pomdp.BeliefSupportWinningRegionQueryInterfaceDouble(model, winning_region)

def build_pomdp(program):
    options = stormpy.BuilderOptions([])
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    options.set_build_all_labels()

    return sp.build_sparse_model_with_options(program, options)

def main():
    random.seed(3)
    parser = argparse.ArgumentParser(description='Starter project for the shielded POMDP simulator.')

    #parser.add_argument('--model', '-m', help='Model file', required=True)
    #parser.add_argument('--property', '-p', help='Property', required=True)

    #args = parser.parse_args()

    path = "/Users/sjunges/cal/gridworld-by-storm/examples/grid_alice_v1.nm"
    annot = dict({'ego-xvar-module' : 'alice',
                      'ego-xvar-name' : 'ax',
                      'ego-yvar-module' : 'alice',
                      'ego-yvar-name': 'ay',
                      'xmax-constant': 'axMax',
                      'ymax-constant': 'ayMax'
                      })
    shield = False
    prism_program = sp.parse_prism_program(path)
    program_annotation = ProgramAnnotation(annot)
    prop = sp.parse_properties_for_prism_program("Pmax=? [ \"notbad\" U \"goal\"]", prism_program)[0]
    raw_formula = prop.raw_formula

    model = build_pomdp(prism_program)
    model = sp.pomdp.make_canonic(model)
    if shield:
        winning_region = compute_winning_region(model, raw_formula)
        otf_shield = construct_otf_shield(model, winning_region)
    else:
        logger.warning("Shielding disabled.")
        otf_shield = NoShield()
    tracker = Tracker(model, otf_shield)


    renderer = Plotter(prism_program,program_annotation,model)
    recorder = VideoRecorder(renderer)
    executor = SimulationExecutor(model, tracker)
    executor.simulate(recorder,nr_runs=1,maxsteps=20)

    renderer.load_ego_image('/Users/sjunges/cal/gridworld-by-storm/tortoise.png', zoom=0.12)
    recorder.save()



if __name__ == "__main__":
    main()
