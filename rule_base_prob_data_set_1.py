import numpy

from GeneticAlgorithm.experiments import run_varied_setting_experiment, \
    write_experiment_data_to_file
from GeneticAlgorithm.common.rbp import BinaryRuleSet, print_rule_set


GRAPHS_DIR = "results/graphs/data1/"
RAW_RESULTS_DIR = "results/raw/data1/"

TRAINING_FILE = "data1.txt"

TRAIN_FROM_RULE = 0
TRAIN_TO_RULE = 32

BASE_CONDITION_SIZE = 5
BASE_RESULT_SIZE = 1
BASE_RULE_COUNT = 10
BASE_RULE_SIZE = BASE_CONDITION_SIZE + BASE_RESULT_SIZE
# Define additional arguments required by the problem specific individual
BASE_INDIV_ADD_STD_ARGS = [
    BASE_CONDITION_SIZE,
    BASE_RESULT_SIZE,
    TRAINING_FILE,
    TRAIN_FROM_RULE,
    TRAIN_TO_RULE,
    True
]

BASE_SETTINGS = {
    "runs": 10,
    "generations": 100,
    "population_size": 250,
    "chromosome_size": BASE_RULE_COUNT * BASE_RULE_SIZE,
    "selection_method": "tournament",
    "tournament_size": 2,
    "single_point_crossover_point": 0.5,
    "mutation_rate_percent": 0.5,
    "indiv_class": BinaryRuleSet,
    "indiv_additional_args": BASE_INDIV_ADD_STD_ARGS
}


def run_wildcard_experiment():
    """Experiment with how using a wildcard makes a difference to fitness"""
    indiv_add_args_wc_disabled = [
        BASE_CONDITION_SIZE,
        BASE_RESULT_SIZE,
        TRAINING_FILE,
        TRAIN_FROM_RULE,
        TRAIN_TO_RULE,
        False
    ]

    exp_title = "Wildcard Enabled Vs Disabled"
    varied_param = "indiv_additional_args"

    exp_settings = dict(BASE_SETTINGS)

    # Include an arg set where wildcards are enabled ano another where they
    # are not enabled.
    exp_settings["indiv_additional_args"] = [
        indiv_add_args_wc_disabled,
        BASE_INDIV_ADD_STD_ARGS
    ]

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


def run_population_size_experiment_with_base_cfg():
    """Experiment with the affects of changing population size when every
    other setting is set to base"""
    exp_title = "Population size parameter"
    varied_param = "population_size"

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["population_size"] = numpy.arange(20, 1000, 20)

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


def run_population_size_experiment_with_0_mutation_per():
    """Experiment with the affects of changing population size when every
    other setting is set to base except mutation which is set 0% of the
    typical range"""
    exp_title = "Population size parameter when mutation is 0% in range"
    varied_param = "population_size"

    num_of_rules = 10

    settings = {
        "runs": 10,
        "generations": 100,
        "population_size": numpy.arange(20, 1000, 20),
        "chromosome_size": num_of_rules * (
                    BASE_CONDITION_SIZE + BASE_RESULT_SIZE),
        "tournament_size": 2,
        "single_point_crossover_point": 0.5,
        "mutation_rate_percent": 0.0,
        "indiv_class": BinaryRuleSet,
        "indiv_additional_args": BASE_INDIV_ADD_STD_ARGS
    }

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        settings,
        varied_param)

    return fittest_individual, stats


def run_population_size_experiment_with_20_mutation_per():
    """Experiment with the affects of changing population size when every
    other setting is set to base except mutation which is set 20% of the
    typical range"""
    exp_title = "Population size parameter when mutation is 20% in range"
    varied_param = "population_size"

    num_of_rules = 10

    settings = {
        "runs": 10,
        "generations": 100,
        "population_size": numpy.arange(20, 1000, 20),
        "chromosome_size": num_of_rules * (
                    BASE_CONDITION_SIZE + BASE_RESULT_SIZE),
        "tournament_size": 2,
        "single_point_crossover_point": 0.5,
        "mutation_rate_percent": 0.2,
        "indiv_class": BinaryRuleSet,
        "indiv_additional_args": BASE_INDIV_ADD_STD_ARGS
    }

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        settings,
        varied_param)

    return fittest_individual, stats


def run_population_size_experiment_with_40_mutation_per():
    """Experiment with the affects of changing population size when every
    other setting is set to base except mutation which is set 40% of the
    typical range"""
    exp_title = "Population size parameter when mutation is 40% in range"
    varied_param = "population_size"

    num_of_rules = 10

    settings = {
        "runs": 10,
        "generations": 100,
        "population_size": numpy.arange(20, 1000, 20),
        "chromosome_size": num_of_rules * (
                    BASE_CONDITION_SIZE + BASE_RESULT_SIZE),
        "tournament_size": 2,
        "single_point_crossover_point": 0.5,
        "mutation_rate_percent": 0.4,
        "indiv_class": BinaryRuleSet,
        "indiv_additional_args": BASE_INDIV_ADD_STD_ARGS
    }

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        settings,
        varied_param)

    return fittest_individual, stats


def run_population_size_experiment_with_60_mutation_per():
    """Experiment with the affects of changing population size when every
    other setting is set to base except mutation which is set 60% of the
    typical range"""
    exp_title = "Population size parameter when mutation is 60% in range"
    varied_param = "population_size"

    num_of_rules = 10

    settings = {
        "runs": 10,
        "generations": 100,
        "population_size": numpy.arange(20, 1000, 20),
        "chromosome_size": num_of_rules * (
                    BASE_CONDITION_SIZE + BASE_RESULT_SIZE),
        "tournament_size": 2,
        "single_point_crossover_point": 0.5,
        "mutation_rate_percent": 0.60,
        "indiv_class": BinaryRuleSet,
        "indiv_additional_args": BASE_INDIV_ADD_STD_ARGS
    }

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        settings,
        varied_param)

    return fittest_individual, stats


def run_population_size_experiment_with_80_mutation_per():
    """Experiment with the affects of changing population size when every
    other setting is set to base except mutation which is set 80% of the
    typical range"""
    exp_title = "Population size parameter when mutation is 80% in range"
    varied_param = "population_size"

    num_of_rules = 10

    settings = {
        "runs": 10,
        "generations": 100,
        "population_size": numpy.arange(20, 1000, 20),
        "chromosome_size": num_of_rules * (
                    BASE_CONDITION_SIZE + BASE_RESULT_SIZE),
        "tournament_size": 2,
        "single_point_crossover_point": 0.5,
        "mutation_rate_percent": 0.80,
        "indiv_class": BinaryRuleSet,
        "indiv_additional_args": BASE_INDIV_ADD_STD_ARGS
    }

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        settings,
        varied_param)

    return fittest_individual, stats


def run_population_size_experiment_with_100_mutation_per():
    """Experiment with the affects of changing population size when every
    other setting is set to base except mutation which is set 100% of the
    typical range"""
    exp_title = "Population size parameter when mutation is 100% in range"
    varied_param = "population_size"

    num_of_rules = 10

    settings = {
        "runs": 10,
        "generations": 100,
        "population_size": numpy.arange(20, 1000, 20),
        "chromosome_size": num_of_rules * (
                    BASE_CONDITION_SIZE + BASE_RESULT_SIZE),
        "tournament_size": 2,
        "single_point_crossover_point": 0.5,
        "mutation_rate_percent": 1,
        "indiv_class": BinaryRuleSet,
        "indiv_additional_args": BASE_INDIV_ADD_STD_ARGS
    }

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        settings,
        varied_param)

    return fittest_individual, stats


def run_single_point_crossover_experiment():
    """Experiment with the affects of changing the single point crossover"""
    exp_title = "Single point crossover variation"
    varied_param = "single_point_crossover_point"

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["single_point_crossover_point"] = [BASE_RULE_SIZE,
                                                    BASE_RULE_SIZE * 2,
                                                    BASE_RULE_SIZE * 3,
                                                    BASE_RULE_SIZE * 4,
                                                    BASE_RULE_SIZE * 5,
                                                    BASE_RULE_SIZE * 6,
                                                    BASE_RULE_SIZE * 7,
                                                    BASE_RULE_SIZE * 8,
                                                    BASE_RULE_SIZE * 9]

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


def run_two_point_crossover_experiment():
    """Experiment with the affects of changing the two step crossover point"""
    exp_title = "Two point crossover variation"
    varied_param = "two_point_crossover_points"

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["two_point_crossover_points"] = [
        [BASE_RULE_SIZE * 1, BASE_RULE_SIZE * 9],  # Crossover rule 2 to rule 9
        [BASE_RULE_SIZE * 2, BASE_RULE_SIZE * 8],  # Crossover rule 3 to rule 8
        [BASE_RULE_SIZE * 3, BASE_RULE_SIZE * 7],  # Crossover rule 4 to rule 7
        [BASE_RULE_SIZE * 4, BASE_RULE_SIZE * 6],  # Crossover rule 5 to rule 6
    ]

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


def run_mutation_only_experiment():
    """Experiment with the affects of changing mutation rate"""
    exp_title = "Mutation rate parameter"
    varied_param = "mutation_rate_percent"

    # Mutation rate typically between 1/pop size and 1/chromosome length.

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["mutation_rate_percent"] = numpy.arange(0.0, 1, 0.1)

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


def run_tournament_size_experiment():
    """Experiment with changing the tournament size"""
    exp_title = "Tournament size variation"
    varied_param = "tournament_size"

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["tournament_size"] = [2, 3, 4, 5, 6, 7, 8]

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


def run_roulette_experiment():
    """Experiment with using roulette wheel selection over tournament"""
    exp_title = "Using roulette wheel selection"
    varied_param = "selection_method"

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["selection_method"] = ["roulette", "tournament"]

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


def run_rule_number_experiment_cfg1():
    """Experiment with what is the minimum amount of rules possible to get max
    fitness, with cfg set 1."""
    exp_title = "Rule base size"
    varied_param = "chromosome_size"

    min_number_of_rules = 2
    max_number_of_rules = 40

    min_chromosome_size = min_number_of_rules * BASE_RULE_SIZE
    max_chromosome_size = max_number_of_rules * BASE_RULE_SIZE
    # Increment two rules at a time in the rule
    inc = BASE_RULE_SIZE * 2

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["population_size"] = 300
    exp_settings["mutation_rate_percent"] = 0.4
    exp_settings["tournament_size"] = 3
    exp_settings["chromosome_size"] = numpy.arange(min_chromosome_size,
                                                   max_chromosome_size,
                                                   inc)

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


def run_rule_number_experiment_cfg2():
    """Experiment with what is the minimum amount of rules possible to get max
    fitness, with cfg set 2."""
    exp_title = "Rule base size cfg 2"
    varied_param = "chromosome_size"

    min_number_of_rules = 2
    max_number_of_rules = 34

    min_chromosome_size = min_number_of_rules * BASE_RULE_SIZE
    max_chromosome_size = max_number_of_rules * BASE_RULE_SIZE
    # Increment two rules at a time in the rule
    inc = BASE_RULE_SIZE * 2

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["population_size"] = 1000
    exp_settings["mutation_rate_percent"] = 0.4
    exp_settings["tournament_size"] = 3
    exp_settings["chromosome_size"] = numpy.arange(min_chromosome_size,
                                                   max_chromosome_size,
                                                   inc)

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


def run_rule_number_experiment_cfg3():
    """Experiment with what is the minimum amount of rules possible to get max
    fitness, with cfg set 3."""
    exp_title = "Rule base size cfg 3"
    varied_param = "chromosome_size"

    min_number_of_rules = 2
    max_number_of_rules = 34

    min_chromosome_size = min_number_of_rules * BASE_RULE_SIZE
    max_chromosome_size = max_number_of_rules * BASE_RULE_SIZE
    # Increment two rules at a time in the rule
    inc = BASE_RULE_SIZE * 2

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["population_size"] = 1000
    exp_settings["mutation_rate_percent"] = 0.4
    exp_settings["tournament_size"] = 3
    exp_settings["chromosome_size"] = numpy.arange(min_chromosome_size,
                                                   max_chromosome_size,
                                                   inc)

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


def run_rule_number_experiment_cfg4():
    """Experiment with what is the minimum amount of rules possible to get max
    fitness, with cfg set 2."""
    exp_title = "Rule base size cfg 4"
    varied_param = "chromosome_size"

    min_number_of_rules = 10
    max_number_of_rules = 12

    min_chromosome_size = min_number_of_rules * BASE_RULE_SIZE
    max_chromosome_size = max_number_of_rules * BASE_RULE_SIZE
    # Increment two rules at a time in the rule
    inc = BASE_RULE_SIZE * 2

    exp_settings = dict(BASE_SETTINGS)
    exp_settings["generations"] = 200
    exp_settings["population_size"] = 400
    exp_settings["mutation_rate_percent"] = 0.4
    exp_settings["tournament_size"] = 3
    exp_settings["chromosome_size"] = numpy.arange(min_chromosome_size,
                                                   max_chromosome_size,
                                                   inc)

    fittest_individual, best_rule_set_settings, stats = run_varied_setting_experiment(
        exp_title,
        exp_settings,
        varied_param)

    return fittest_individual, stats


if __name__ == '__main__':
    experiment_stats = []
    # Stores all the tests to be executed
    enabled_experiments = [
        #    run_wildcard_experiment,
        #    run_rule_number_experiment_cfg4
        #    run_rule_number_experiment_cfg1
        #    run_roulette_experiment
        #    run_tournament_size_experiment,
        #    run_mutation_only_experiment,
        #    run_population_size_experiment_with_base_cfg,
        #    run_single_point_crossover_experiment,
        #    run_two_point_crossover_experiment,
        #    run_population_size_experiment_with_80_mutation_per,
        #    run_population_size_experiment_with_100_mutation_per
        #    run_population_size_experiment_with_75_mutation_per
    ]

    # Execute each enabled test
    for run_exp in enabled_experiments:
        exp_fittest_indiv, exp_stats = run_exp()

        # experiment_stats.append(exp_stats)

        print("Fittest rule set:")
        print_rule_set(exp_fittest_indiv, BASE_CONDITION_SIZE, BASE_RESULT_SIZE)

        # Write raw data to a CSV file
        write_experiment_data_to_file([exp_stats], RAW_RESULTS_DIR)

    # Produce a graph for each result


#  graph_varied_setting_experiments(experiment_stats, GRAPHS_DIR)
