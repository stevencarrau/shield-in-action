import argparse

import stormpy as sp


def check(path_to_model, property_str):
    program = sp.parse_prism_program(path_to_model)
    props = sp.parse_properties(property_str, program)
    model = sp.build_model(program, props)
    result = sp.model_checking(model, props[0])
    return result.at(model.initial_states[0]) > 0.5


def main():
    parser = argparse.ArgumentParser(description='Starter project for stormpy.')

    parser.add_argument('--model', '-m', help='Model file', required=True)
    parser.add_argument('--property', '-p', help='Property', required=True)
    args = parser.parse_args()

    # Call function
    res = check(args.model, args.property)
    print("Result > 0.5? " + ("yes" if res else "no"))


if __name__ == "__main__":
    main()
