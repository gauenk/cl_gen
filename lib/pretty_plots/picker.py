from inquirer import List,prompt
import pretty_plots.comparing_denoiser_quality
import pretty_plots.example_denoised_images
import pretty_plots.nnf_v_of
import pretty_plots.example_bootstrapping
import pretty_plots.stat_test_properties

def run_picker():
    options = [
        List('plot',
                      message="Pick a plot to create",
                      choices=["stat_test_properties",
                               "example_bootstrapping",
                               "nnf_v_of",
                               "comparing_denoiser_quality",
                               "example_denoised_images",
                               "none"]
        )
    ]
    answer = prompt(options)
    if answer['plot'] == "comparing_denoiser_quality":
        comparing_denoiser_quality.run()
    elif answer['plot'] == "example_denoised_images":
        example_denoised_images.run()
    elif answer['plot'] == "nnf_v_of":
        nnf_v_of.run()
    elif answer['plot'] == "example_bootstrapping":
        example_bootstrapping.run()
    elif answer['plot'] == "stat_test_properties":
        stat_test_properties.run()
    elif answer['plot'] == "none":
        print("No plot today! Go outside.")

