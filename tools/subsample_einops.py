# -- python imports
import pdb
import numpy as np
from einops import rearrange, reduce
import matplotlib.pyplot as plt

import tkinter as tk
from PIL import ImageTk, Image

def show_image(pics):
    formatted = (pics * 255 / np.max(pics)).astype('uint8')
    image_window = tk.Tk()
    img = ImageTk.PhotoImage(Image.fromarray(formatted))
    panel = tk.Label(image_window, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")
    image_window.mainloop()

def main():
    pics = np.load('./data/test_images.npy',allow_pickle=False)
    import IPython; IPython.embed(); exit(1)
    pics = rearrange(pics,'b h w c -> h (b w) c')

    

if __name__ == "__main__":
    print("HI")
    main()

