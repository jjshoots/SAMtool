from PIL import Image, ImageTk
import argparse
import os
import sys
import tkinter
from samtool.sammer import FileSeeker

def click():
    pass

def create_app(imagedir, labeldir, annotations):
    seeker = FileSeeker(imagedir, labeldir, annotations)

    # create root window
    root = tkinter.Tk()
    root.title("SAMTool Tkinter GUI")
    root.geometry("1920x1080")

    # all buttons
    button_prev_unlabelled = tkinter.Button(root, text="Prev Unlabelled", bg="orange", command=click)
    button_prev = tkinter.Button(root, text="Previous", bg="grey", command=click)
    button_next = tkinter.Button(root, text="Next", bg="grey", command=click)
    button_next_unlabelled = tkinter.Button(root, text="Next Unlabelled", bg="orange", command=click)

    button_prev_unlabelled.grid(column=1, row=0)
    button_prev.grid(column=2, row=0)
    button_next.grid(column=3, row=0)
    button_next_unlabelled.grid(column=4, row=0)

    button_reset_selection = tkinter.Button(root, text="Reset Selection", bg="grey", command=click)
    button_reset_label = tkinter.Button(root, text="Reset Label", bg="grey", command=click)
    button_reset_all = tkinter.Button(root, text="Reset All", bg="grey", command=click)

    button_reset_selection.grid(column=1, row=1)
    button_reset_label.grid(column=2, row=1)
    button_reset_all.grid(column=3, row=1)

    button_approve = tkinter.Button(root, text="Approve", bg="orange", command=click)
    button_negate = tkinter.Button(root, text="Negate", bg="grey", command=click)

    button_approve.grid(column=1, row=2)
    button_negate.grid(column=2, row=2)

    # image displays
    imagepath = seeker.file_increment(ascend=True, unlabelled_only=True, filename="")
    imagepath = os.path.join(imagedir, imagepath)
    image = Image.open(imagepath)
    img = ImageTk.PhotoImage(image.resize(tuple(int(size / 2) for size in image.size[:2])))
    display_partial = tkinter.Label(root, image=img)
    display_partial.grid(column=1, row=3)

    # Execute Tkinter
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        prog="SAMTool QT",
        description="Semantic Segmentation Dataset Creation Tool powered by Segment Anything Model from Meta.",
    )
    parser.add_argument("--imagedir", required=True)
    parser.add_argument("--labeldir", required=True)
    parser.add_argument("--annotations", required=True)
    args = parser.parse_args()

    create_app(args.imagedir, args.labeldir, args.annotations)
