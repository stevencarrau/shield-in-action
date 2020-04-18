import argparse
import random

import stormpy as sp
import stormpy.examples
import stormpy.examples.files
import stormpy.simulator
import stormpy.pomdp

import gridstorm.plotter as plotter

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)

class Tracker:
    def __init__(self, model, shield):
        self._model = model
        self._tracker = stormpy.pomdp.BeliefSupportTrackerDouble(model)
        self._shield = shield

    def track(self, action, observation):
        logger.debug(f"Track action={action}, observation={observation}")
        self._tracker.track(action, observation)

    def monitor(self):
        return self._shield.query_current_belief(self._tracker.get_current_belief_support())

    def shielded_actions(self, action_indices):
        safe_action_indices = []
        for a in action_indices:
            if self._shield.query_action(self._tracker.get_current_belief_support(), a):
                safe_action_indices.append(a)
        return safe_action_indices



    def list_support(self):
        return [s for s in self._tracker.get_current_belief_support()]

    def reset(self):
        self._tracker = stormpy.pomdp.BeliefSupportTrackerDouble(self._model)


def simulate(model, shield, plotter = None):
    simulator = stormpy.simulator.create_simulator(model, seed=42)
    simulator.set_full_observability(True)
    tracker = Tracker(model, shield)

    # 5 paths of at most 20 steps.
    paths = []
    observed_paths = []
    for m in range(1):
        state = simulator.restart()
        tracker.reset()
        path_str = [f"{state}"]
        path = [state]
        #if plotter:
        #    plotter.render(state, tracker.list_support())

        observed_path_str = [f"{tracker.list_support()}"]
        observed_path = [tracker.list_support()]
        for n in range(30):
            actions = simulator.available_actions()
            safe_actions = tracker.shielded_actions(range(len(actions)))
            logger.debug(f"Number of actions: {actions}. Safe action indices: {safe_actions}")
            select_action = random.randint(0, len(actions) - 1)
            action = actions[select_action]
            # print(f"Randomly select action nr: {select_action} from actions {actions}")
            path_str.append(f"--act={action}-->")
            observed_path_str.append(f"--act={action}-->")
            state = simulator.step(actions[select_action])
            tracker.track(select_action, model.get_observation(state))
            # print(state)
            assert state in tracker.list_support()
            path_str.append(f"{state}")
            observed_path_str.append(f"{tracker.list_support()}")
            path.append(state)
            observed_path.append(tracker.list_support())
            logger.debug(f"Now in state {state}. Belief: {tracker.list_support()}. Safe: {tracker.monitor()}")
            #if plotter:
            #    plotter.render(state, tracker.list_support())

            if simulator.is_done():
                logger.debug(f"Done!")
                # print("Trapped!")
                break
        paths.append(path)
        observed_paths.append(observed_path)
        mp4file = f"test-run{m}.mp4"
        plotter.record(mp4file, path, observed_path)
    for path in paths:
        print(" ".join(path))
    for observed_path in observed_paths:
        print(" ".join(observed_path))

def compute_winning_region(model, formula):
    options = sp.pomdp.IterativeQualitativeSearchOptions()
    solver = sp.pomdp.create_iterative_qualitative_search_solver_Double(model, formula, options)
    ## TODO select a good lookahead
    solver.compute_winning_region(model.nr_states)
    return solver.last_winning_region

def construct_otf_shield(model, winning_region):
    return sp.pomdp.BeliefSupportWinningRegionQueryInterfaceDouble(model, winning_region)

def main():
    parser = argparse.ArgumentParser(description='Starter project for stormpy.')

    #parser.add_argument('--model', '-m', help='Model file', required=True)
    #parser.add_argument('--property', '-p', help='Property', required=True)

    #args = parser.parse_args()

    path = "/Users/sjunges/cal/gridworld-by-storm/examples/grid_alice_v1.nm"
    prism_program = sp.parse_prism_program(path)
    prop = sp.parse_properties_for_prism_program("Pmax=? [ \"notbad\" U \"goal\"]", prism_program)[0]
    raw_formula = prop.raw_formula

    options = stormpy.BuilderOptions([])
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    options.set_build_all_labels()

    model = sp.build_sparse_model_with_options(prism_program, options)
    state_vals = model.state_valuations
    model = sp.pomdp.make_canonic(model)
    # TODO make cannoci shoudl prserve state labels
    winning_region = compute_winning_region(model, raw_formula)
    otf_shield = construct_otf_shield(model, winning_region)

    import plotter
    plotter = plotter.Plotter(prism_program, model, state_vals, 6,6)

    simulate(model, otf_shield, plotter)
    print(model)


if __name__ == "__main__":
    main()
