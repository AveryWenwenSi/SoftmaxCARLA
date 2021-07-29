import argparse
import logging

from agents.imitation.imitation_learning import ImitationLearning

import tensorflow as tf
import glob
from PIL import Image
from numpy import asarray, save

if (__name__ == '__main__'):
    agent = ImitationLearning()
    
    data_paths = "./adverse_drive_data/"
    data_dirs = ["out_example_episode/out_images/", "training/right_clearnoon/", "training/right_cloudynoon/", ]
    control_input = 0

    for i, p in enumerate(data_dirs):
        outputs = []
        data_dir = data_paths + p + "/*.png"
        images = glob.glob(data_dir)
        for image in images:
            rgb_image = asarray(Image.open(image))
            steers = agent.run_step(rgb_image, control_input)
            outputs.append(steers)
        outputs = asarray(outputs)
        with open("test"+str(i)+".npy", 'wb') as f:
            save(f, outputs[:60,0])






