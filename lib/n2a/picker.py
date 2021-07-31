from inquirer import List,prompt
import n2a.exps.unsup
import n2a.exps.nnf

def run_experiment_picker():
    options = [
        List('experiment',
                      message="Pick an experiment to run",
                      choices=["NNF Quality",
                               "Unsupervised Burst Denoising Quality",
                               "none"
                      ]
        )
    ]
    answer = prompt(options)
    if answer['experiment'] == "NNF Quality":
        nnf.run()
    elif answer['experiment'] == "Unsupervised Burst Denoising Quality":
        unsup.run()
    elif answer['experiment'] == "none":
        print("No plot today! Go outside :D")

