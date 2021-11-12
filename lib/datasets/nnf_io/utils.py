
from PIL import Image


def read_ishape(frame_path):
    w,h = Image.open(frame_path).size
    ishape = (h,w,3)
    return ishape

