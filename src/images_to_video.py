# Inspired by https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python

import os
import moviepy.video.io.ImageSequenceClip
import argparse

parser = argparse.ArgumentParser(description="Converts a collection of images to video")
parser.add_argument("images_directory", type=str, help="Path to images directory")
parser.add_argument("fps", type=int, help="FPs")

args = parser.parse_args()

image_files = [
    os.path.join(args.images_directory, img)
    for img in os.listdir(args.images_directory)
    if img.endswith(".png")
]
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=args.fps)
clip.write_videofile(args.images_directory + "/sim_video.mp4")
