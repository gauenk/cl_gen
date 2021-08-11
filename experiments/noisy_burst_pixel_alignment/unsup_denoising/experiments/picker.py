from inquirer import List,prompt

import unsup_denoising.experiments.compare_to_competitors as compare_to_competitors

def run():
    options = [
        List('exp',
             message="Pick a experiment of [Noisy Alignment] to run!",
             choices=["compare_to_competitors",
                      "none"]
        )
    ]
    answer = prompt(options)

    if answer['exp'] == "compare_to_competitors":
        info = compare_to_competitors.get_run_info()
        return info
    elif answer['exp'] == "none":
        print("No experiment today! Go outside :D")

def get_all_exps():
    all_info = [compare_to_competitors.get_run_info(),
    ]
    return all_info
