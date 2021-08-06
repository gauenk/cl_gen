from inquirer import List,prompt
# import noisy_burst_pixel_alignment.noisy_alignment
# import noisy_burst_pixel_alignment.unsup_denoising_dl
# import noisy_burst_pixel_alignment.unsup_denoising_cl
# import noisy_burst_pixel_alignment.sup_denoising
# import noisy_burst_pixel_alignment.noisy_hdr

import noisy_alignment
import unsup_denoising
import sup_denoising
import noisy_hdr

def run():
    options = [
        List('exp',
             message="Pick a experiment to run!",
             choices=["noisy_alignment",
                      "sup_denoising",
                      "unsup_denoising",
                      "noisy_hdr",
                      "none"]
        )
    ]
    answer = prompt(options)
    choices=["noisy_alignment",
             "sup_denoising",
             "unsup_denoising",
             "noisy_hdr",
             "none"]

    if answer['exp'] == "noisy_hdr":
        noisy_hdr.run()
    elif answer['exp'] == "sup_denoising":
        sup_denoising.run()
    elif answer['exp'] == "unsup_denoising":
        unsup_denoising.run()
    elif answer['exp'] == "noisy_alignment":
        noisy_alignment.run()
    elif answer['exp'] == "none":
        print("No experiment today! Go outside :D")

