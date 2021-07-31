from inquirer import List,prompt

import noisy_alignment.experiments.compare_to_theory as compare_to_theory
import noisy_alignment.experiments.compare_to_competitors as compare_to_competitors
import noisy_alignment.experiments.stress_tests as tress_tests

def run():
    options = [
        List('exp',
             message="Pick a experiment of [Noisy Alignment] to run!",
             choices=["compare_to_theory",
                      "compare_to_competitors",
                      "stress_tests",
                      "none"]
        )
    ]
    answer = prompt(options)

    if answer['exp'] == "compare_to_theory":
        info = compare_to_theory.get_run_info()
        return info
    elif answer['exp'] == "compare_to_competitors":
        info = compare_to_competitors.get_run_info()
        return info
    elif answer['exp'] == "stress_tests":
        info = stress_tests.get_run_info()
        return info
    elif answer['exp'] == "none":
        print("No experiment today! Go outside :D")

def get_all_exps():
    all_info = [compare_to_theory.get_run_info(),
                compare_to_competitors.get_run_info(),
                stress_tests.get_run_info()
    ]
    return all_info
