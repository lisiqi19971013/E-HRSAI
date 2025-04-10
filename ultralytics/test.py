import glob
import os
import shutil
import cv2
from sympy.core.random import shuffle
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--ipFolder", type=str, default='../output_main')
parser.add_argument("--opFolder", type=str, default='../detect_result')
parser.add_argument("--ckptPath", type=str, default='./yolov8x.pt')
args = parser.parse_args()

model = YOLO(args.ckptPath)

input_folder = args.ipFolder
opFolder = args.opFolder

os.makedirs(opFolder, exist_ok=True)

img_list = glob.glob(os.path.join(input_folder, '*_output.png'))

for i in range(len(img_list)):
    results = model.predict(source=img_list[i], save=False)

    result = results[0]

    plotted_img = result.plot(
                    line_width=6,
                    font_size=5,
                    boxes=True,
                    conf=True,
                    labels=True,
                    im_gpu=None,
                )

    cv2.imwrite(f'{opFolder}/{os.path.split(img_list[i])[-1]}', plotted_img)