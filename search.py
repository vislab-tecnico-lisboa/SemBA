#################################################################################################################
# Visual Search: Semantic-Based Bayesian Attention (SemBA) x Multi-Scale Fovea (MS-Fov)                         #  
# Joao Luzio, Institute for Systems and Robotics, Técnico Lisboa, 2026                                          #  
# example usage (1680x1050 image): python search.py -f examples/bottle.jpg -t bottle -d dfine -l 4 -b 160       #
#################################################################################################################

import argparse
import numpy as np
import skimage.io
import time
import random
import os

# Multi-Scale Fovea
from foveation import MS_Foveation

# Object Detection Models
import utils.detectors as detect

# Semantic Data Fusion
import utils.semba as semba

# Generic Functions
import utils.general as utils

# Import Configurations
from utils.configs import *

# Argument Parser
parser = argparse.ArgumentParser(
    description='Visual Search: Semantic-Based Bayesian Attention (SemBA) x Multi-Scale Fovea (MS-Fov)')
parser.add_argument('-f','--file_name', type=str, default='examples/bottle.jpg', required=False)
parser.add_argument('-d','--detector', type=str, default='dfine', required=False)
parser.add_argument('-t','--category', type=str, default='bottle', required=False)
parser.add_argument('-l','--levels', type=int, default=4, required=False)
parser.add_argument('-b','--base_dim', type=int, default=160, required=False)

#################################################################################################################
# Visual Target Search                                                                                          #
#################################################################################################################

def main():

    # argument parsing
    args = parser.parse_args()
    try:
        assert args.detector in DETECTORS, f"Unknown object detection model. Available options are: {DETECTORS}"
        assert args.category in CLASS_NAMES, f"Unknown target class. Available options are:\n{CLASS_NAMES[1:]}"
        assert args.base_dim >= 128, "Invalid base layer dimension. The value must be greater or equal to 128."
        assert args.levels >= 1, "Invalid mumber of fovea levels. The value must be greater or equal to 1."
        assert os.path.exists(args.file_name), f"The file {args.file_name} does not exist."
        if args.detector == 'detr': is_detr = True
        else: is_detr = False
    except AssertionError as error:
        print(error)
        exit(-1)

    # semantic maps (rows, columns)
    map_dim = (Y_CELLS,X_CELLS)
    
    # Force IOR True
    ior = True

    print("\nDeep Object Detector : {}".format(args.detector))
    print("\nMulti-Scale Fovea Dimensions : {}x{}x{}".format(args.levels, args.base_dim, args.base_dim))
    print("\nInhibition of Return: {}\n".format(ior))

    # Multi-Scale Fovea definition
    foveator = MS_Foveation(args.levels, args.base_dim, scale_factor=2)

    # Deep Object Detection Model
    model, processor = detect.load_model(args.detector, args.base_dim)
    total_classes = len(CLASS_NAMES)-1
    score_thres = CONF_THRESH

    # download image (scene)
    try:
        img = skimage.io.imread(args.file_name) 
        if img is not None:
            height, width, _ = img.shape
        else:
            raise ValueError
    except:
        raise FileNotFoundError
    print(f'\n\nFile Name: {args.file_name}')
    print('Image dims (height, width): {}, {}'.format(height, width))

    # get the center for generated for this image
    xm, ym = int(X_CELLS/2), int(Y_CELLS/2)
    map_inhib_ret = np.zeros(map_dim)
    center = np.array([int(width/2),int(height/2)])
        
    # semba map fusion variables
    map = np.ones((map_dim[0], map_dim[1], total_classes))
    attention_maps, ctrs = [], []

    # target definition and success rate computation setup
    target_cls = CLASS_NAMES.index(args.category)
    time_count = 0.0

    print('Target Class: {} ({})\n\n'.format(CLASS_NAMES[target_cls],target_cls))
    try:

        for fixations in range(MAX_FIX+1):
                    
            print('\n\nFocal point {}: [{},{}] -> {}\n\n'.format(fixations,xm,ym,center))
            start = time.time()

            # foveate image around a center point
            layers = foveator.foveate(img, center)
            ctrs.append({'X': center[0], 'Y': center[1]})

            # generate object predictions
            pred = detect.predict(layers, model, processor, score_thres=score_thres, is_detr=is_detr)

            # fuse categorical information
            for lvl in range(args.levels):

                det = np.array(pred[lvl]) 
                det_scores = semba.fov_observation_model(det,total_classes)

                if pred[lvl] != []: # Check if there are any detections at all

                    for i in range(det.shape[0]):
                                
                        # scale and position each bounding-box on the original image
                        bbox = det[i,0:4]
                        box = foveator.bbox_remapping(bbox, lvl+1, center)

                        # cell-by-cell fusion
                        for y in range(map_dim[0]):
                            for x in range(map_dim[1]):
                                if utils.in_cell(y,x,box,map_dim,height,width):
                                    map[y,x] = semba.fusion_model(map[y,x], det_scores[i])                      

                else: # avoid 'stuck state'

                    utils.ior_in_area(map_inhib_ret, ym, xm, map_dim)

            # active perception
            target_map = np.zeros(map_dim)
            for y in range(Y_CELLS):
                for x in range(X_CELLS):
                    target_map[y,x] = map[y,x,target_cls-1]/np.sum(map[y,x])

            # termination condition
            if target_map[ym,xm] >= TERMINATION_THRESH: break

            attention_maps.append(np.copy(target_map))

            # determine the next best focal point 
            while(True):
                idx = np.argwhere(target_map == np.amax(target_map))
                rand_val = random.randint(0,idx.shape[0]-1)
                ym, xm = idx[rand_val,0], idx[rand_val,1]
                if map_inhib_ret[ym,xm] == 0:
                    if ior: utils.ior_in_area(map_inhib_ret, ym, xm, map_dim)
                    break
                else:
                    target_map[ym,xm] = 0.0

            # define estimated center in center of the cell
            ctr = utils.cell_center(ym, xm, map_dim, height, width)
            center = np.array([int(ctr[1]),int(ctr[0])])
                
            end = time.time()
            print("Time elapsed during active perception: {:.2f} seconds\n".format(end-start))
            time_count += end-start

    except KeyboardInterrupt:

        print('\nInterrupted!\n')

    finally:

        avg_time = time_count/(fixations+1)

    print("\nAverage time per fixation: {:.2f} seconds\n".format(avg_time))

    # create a directgory to store the results
    path = utils.create_dir("runs/")

    # generate attention maps and gif
    for i in range(len(attention_maps)): utils.save_map(attention_maps[i], path, i)
    utils.generate_gif(path,fps=2) 

    # plot scanpath
    xs = np.array([c['X'] for c in ctrs])
    ys = np.array([c['Y'] for c in ctrs])
    utils.plot_scanpath(img, xs, ys, file_name=path+"/scanpath.png")

if __name__ == "__main__":
    main()