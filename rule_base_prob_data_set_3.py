import numpy

from GeneticAlgorithm.experiments import write_experiment_data_to_file, \
    run_varied_setting_experiment

from GeneticAlgorithm.common.rbp import FloatRuleSet, print_rule_set, \
    print_rule_set_with_tolerance, test_rule_set

GRAPHS_DIR = "results/graphs/data3/"
RAW_RESULTS_DIR = "results/raw/data3/"


TRAINING_FILE = "data3.txt"
MAX_RULE_NUMBER = 2000


# Condition is * 2 of variable size so that ....
CONDITION_SIZE = 14
RESULT_SIZE = 1
RULE_SIZE = CONDITION_SIZE + RESULT_SIZE


BASE_RULE_COUNT = 10
BASE_TRAIN_FROM_RULE = 1
BASE_TRAIN_TO_RULE = 1000
BASE_TEST_FROM_RULE = 1001
BASE_TEST_TO_RULE = 2000
# The upper and lower bound a gene could mutate by
BASE_MUTATION_CREEP = 0.1

# Define additional arguments required by the problem specific individual
BASE_INDIV_ADD_ARGS = [
    CONDITION_SIZE,
    RESULT_SIZE,
    BASE_MUTATION_CREEP,
    TRAINING_FILE,
    BASE_TRAIN_FROM_RULE,
    BASE_TRAIN_TO_RULE,
    TRAINING_FILE,
    BASE_TEST_FROM_RULE,
    BASE_TEST_TO_RULE
]

BASE_SETTINGS = {
    "runs": 5,
    "generations": 100,
    "population_size": 200,
    "chromosome_size": BASE_RULE_COUNT * RULE_SIZE,
    "selection_method": "tournament",
    "tournament_size": 2,
    "single_point_crossover_point": 0.5,
    "mutation_rate_percent": 0.5,
    "indiv_class": FloatRuleSet,
    "indiv_additional_args": BASE_INDIV_ADD_ARGS
}


def run_training_rule_set_size():
    exp_title = "Training and test data distribution"
    varied_param = "indiv_additional_args"

    arg_sets = []

    # What ever rules aren't used for training are used for testing.
    # Test size = maximum rules in file - training size
    training_sizes = [600, 800, 1000, 1200, 1400, 1600]

    # Create an argument set for each training and testing size
    for training_size in training_sizes:
        # Training rules will always start from 0 and go up to a set size
        arg_sets.append([
            CONDITION_SIZE,
            RESULT_SIZE,
            BASE_MUTATION_CREEP,
            TRAINING_FILE,
            BASE_TRAIN_FROM_RULE,
            training_size,
            TRAINING_FILE,
            training_size+1,
            BASE_TEST_TO_RULE
        ])

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["indiv_additional_args"] = arg_sets
    # exp_settings["population_size"] = 400

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param,
        test_func=test_rule_set)

    return fittest_individual, stats


def run_mutation_creep_experiment():
    """Experiment with changing the creep value"""
    exp_title = "Mutation creep experiment"
    varied_param = "indiv_additional_args"

    arg_sets = []

    for creep_val in numpy.arange(0.05, 0.3, 0.05):
        arg_sets.append([
            CONDITION_SIZE,
            RESULT_SIZE,
            creep_val,
            TRAINING_FILE,
            BASE_TRAIN_FROM_RULE,
            BASE_TRAIN_TO_RULE,
            TRAINING_FILE,
            BASE_TEST_FROM_RULE,
            BASE_TEST_TO_RULE
        ])

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["runs"] = 5
    exp_settings["indiv_additional_args"] = arg_sets

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


def run_population_size_experiment():
    """Experiment with the affects of changing population size when every
    other setting is set to base"""
    exp_title = "Population size parameter"
    varied_param = "population_size"

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["runs"] = 5
    exp_settings["population_size"] = numpy.arange(100, 1000, 200)

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param,
        test_func=test_rule_set)

    return fittest_individual, stats


def run_rule_number_experiment_cfg1():
    """Experiment with what is the minimum amount of rules possible to get max
    fitness, with cfg set 1."""
    exp_title = "Rule base size 1 cfg P1"
    varied_param = "chromosome_size"

    min_number_of_rules = 8
    max_number_of_rules = 10

    min_chromosome_size = min_number_of_rules * RULE_SIZE
    max_chromosome_size = max_number_of_rules * RULE_SIZE
    # Increment two rules at a time in the rule
    inc = RULE_SIZE

    mutation_creep = 0.45

    args = [
            CONDITION_SIZE,
            RESULT_SIZE,
            mutation_creep,
            TRAINING_FILE,
            BASE_TRAIN_FROM_RULE,
            BASE_TRAIN_TO_RULE,
            TRAINING_FILE,
            BASE_TEST_FROM_RULE,
            BASE_TEST_TO_RULE
    ]

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["population_size"] = 500
    exp_settings["generations"] = 150
    exp_settings["runs"] = 3
    exp_settings["mutation_rate_percent"] = 0.5
    exp_settings["indiv_additional_args"] = args
    exp_settings["chromosome_size"] = numpy.arange(min_chromosome_size,
                                                   max_chromosome_size,
                                                   inc)

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param,
        test_func=test_rule_set)

    return fittest_individual, stats

if __name__ == '__main__':
    experiment_stats = []
    # Stores all the tests to be executed
    enabled_experiments = [
        # run_rule_number_experiment_cfg1
        # run_training_rule_set_size,
        # run_population_size_experiment,
        # run_population_size_experiment,
    ]

    # Execute each enabled test
    for run_exp in enabled_experiments:
        exp_fittest_indiv, exp_stats = run_exp()

        # experiment_stats.append(exp_stats)

        print("Fittest rule set:")
        print_rule_set_with_tolerance(exp_fittest_indiv, CONDITION_SIZE,
                                      RESULT_SIZE)

        # Write raw data to a CSV file
        write_experiment_data_to_file([exp_stats], RAW_RESULTS_DIR)