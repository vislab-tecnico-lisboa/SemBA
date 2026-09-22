"""
Visual search with Semantic-Based Bayesian Attention and Multi-Scale Fovea.
Joao Luzio, Institute for Systems and Robotics, Técnico Lisboa, 2026.
Example: python search.py -f examples/bottle.jpg -t bottle -d dfine -l 4 -b 160
"""

import argparse
from pathlib import Path
import random
import time

import numpy as np
import skimage.io

from foveation import MS_Foveation
from utils import general as utils
from utils import semba
from utils.configs import (
    CLASS_NAMES, CONF_THRESH, DETECTORS, MAX_FIX, TERMINATION_THRESH,
    X_CELLS, Y_CELLS,
)


def parse_args(argv=None):
    """Parse and validate settings before loading the detection model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f', '--file_name', default='examples/bottle.jpg')
    parser.add_argument('-d', '--detector', choices=sorted(DETECTORS), default='dfine')
    parser.add_argument('-t', '--category', choices=CLASS_NAMES[1:], default='bottle')
    parser.add_argument('-l', '--levels', type=int, default=4)
    parser.add_argument('-b', '--base_dim', type=int, default=160)
    args = parser.parse_args(argv)
    if args.levels < 1:
        parser.error('The number of fovea levels must be at least 1.')
    min_dim = 64 if args.detector == 'detr' else 128
    if args.base_dim < min_dim:
        parser.error(f'The base layer dimension must be at least {min_dim}.')
    if not Path(args.file_name).is_file():
        parser.error(f'The file {args.file_name} does not exist.')
    return args


def overlapping_cells(box, x_edges, y_edges):
    """Match in_cell's inclusive-pixel intersection convention on the grid."""
    x_overlap = np.minimum(box[2], x_edges[1:]) - np.maximum(box[0], x_edges[:-1]) + 1
    y_overlap = np.minimum(box[3], y_edges[1:]) - np.maximum(box[1], y_edges[:-1]) + 1
    return (y_overlap[:, None] > 0) & (x_overlap[None, :] > 0)


def next_fixation(target_map, inhibited):
    """Choose uniformly among the best available cells, or return None."""
    available = ~inhibited.astype(bool)
    if not available.any():
        return None
    best_score = target_map[available].max()
    candidates = np.argwhere(available & (target_map == best_score))
    return tuple(candidates[random.randrange(len(candidates))])


def main(argv=None):
    
    args = parse_args(argv)
    image = skimage.io.imread(args.file_name)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('The scene must be an RGB image with three channels.')
    height, width = image.shape[:2]

    # Delay heavyweight detector imports until arguments and image are valid.
    from utils import detectors as detect

    foveator = MS_Foveation(args.levels, args.base_dim, scale_factor=2)
    model, processor = detect.load_model(args.detector, args.base_dim)
    map_shape = (Y_CELLS, X_CELLS)
    total_classes = len(CLASS_NAMES) - 1
    target_class = CLASS_NAMES.index(args.category)
    beliefs = np.ones((*map_shape, total_classes))
    inhibited = np.zeros(map_shape, dtype=bool)

    # Multiplication matches the original cell boundary arithmetic.
    x_edges = np.arange(X_CELLS + 1) * (width / X_CELLS)
    y_edges = np.arange(Y_CELLS + 1) * (height / Y_CELLS)
    row, column = Y_CELLS // 2, X_CELLS // 2
    center = np.array([width // 2, height // 2])
    attention_maps, centers, durations = [], [], []

    print(f'\nDeep Object Detector: {args.detector}')
    print(f'Multi-Scale Fovea Dimensions: {args.levels}x{args.base_dim}x{args.base_dim}')
    print(f'Inhibition of Return: True\nFile Name: {args.file_name}')
    print(f'Image dims (height, width): {height}, {width}')
    print(f'Target Class: {args.category} ({target_class})')

    try:
        # Preserve the initial fixation plus MAX_FIX subsequent fixations.
        for fixation in range(MAX_FIX + 1):
            print(f'\nFocal point {fixation}: [{column},{row}] -> {center}')
            start = time.perf_counter()
            layers = foveator.foveate(image, center)
            centers.append(center.copy())
            predictions = detect.predict(
                layers, model, processor, score_thres=CONF_THRESH,
                is_detr=args.detector == 'detr',
            )
            for level, detections in enumerate(predictions, start=1):
                if len(detections) == 0:
                    utils.ior_in_area(inhibited, row, column, map_shape)
                    continue
                detections = np.asarray(detections)
                scores = semba.fov_observation_model(detections, total_classes)
                for detection, observation in zip(detections, scores):
                    box = foveator.bbox_remapping(detection[:4].copy(), level, center)
                    mask = overlapping_cells(box, x_edges, y_edges)
                    beliefs[mask] = semba.fusion_model(beliefs[mask], observation)

            target_map = semba.attention_map(beliefs, map_shape, target_class)
            attention_maps.append(target_map.copy())
            found = target_map[row, column] >= TERMINATION_THRESH
            next_cell = None
            if not found and fixation < MAX_FIX:
                next_cell = next_fixation(target_map, inhibited)
                if next_cell is not None:
                    row, column = next_cell
                    utils.ior_in_area(inhibited, row, column, map_shape)
                    cy, cx = utils.cell_center(row, column, map_shape, height, width)
                    center = np.array([int(cx), int(cy)])
            durations.append(time.perf_counter() - start)
            print(f'Time elapsed during active perception: {durations[-1]:.2f} seconds')
            if found or next_cell is None:
                break
    except KeyboardInterrupt:
        print('\nInterrupted!')

    average_time = sum(durations) / len(durations) if durations else 0.0
    print(f'\nAverage time per fixation: {average_time:.2f} seconds')
    path = Path(utils.create_dir('runs/'))
    for index, attention in enumerate(attention_maps):
        utils.save_map(attention, str(path), index)
    if attention_maps:
        utils.generate_gif(str(path), fps=2, image=image,
                           attention_maps=attention_maps, fixations=centers)
    if centers:
        coordinates = np.asarray(centers)
        utils.plot_scanpath(image, coordinates[:, 0], coordinates[:, 1],
                            file_name=str(path / 'scanpath.png'))


if __name__ == '__main__':
    main()
