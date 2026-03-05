###############################################################################################
# General Functions for SemBA x MS-Fovea                                                      #
###############################################################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imageio
import os
import re

# create a new directory to save a new run
def create_dir(base_folder):

    os.makedirs(base_folder, exist_ok=True)

    pattern = re.compile(r"run_(\d+)$")
    existing_k = []

    for name in os.listdir(base_folder):
        path = os.path.join(base_folder, name)

        if os.path.isdir(path):
            match = pattern.match(name)
            if match:
                existing_k.append(int(match.group(1)))

    # Determine next k
    next_k = max(existing_k, default=0) + 1

    new_folder = os.path.join(base_folder, f"run_{next_k}")
    os.makedirs(new_folder)

    print(f"\nResults for this run ({next_k}) are saved in: {new_folder}\n")

    return new_folder

# save attention map
def save_map(map, path, idx):
    plt.imshow(map)
    plt.axis('off')
    plt.savefig(path+f"/map_fix{idx}.png",bbox_inches='tight',pad_inches=0,dpi=400)    
    plt.clf() 

# print scanpath on top of the original image
def plot_scanpath(img, xs, ys, file_name="scanpath.png", title=None):
    fig, ax = plt.subplots()
    ax.imshow(img)

    for i in range(len(xs)):
        if i > 0:
            plt.arrow(xs[i - 1], ys[i - 1],
                        xs[i] - xs[i - 1],
                        ys[i] - ys[i - 1],
                        width=6,
                        color='yellow',
                        alpha=0.8)

    for i in range(len(xs)):

        cir_rad = 30
        if i == len(xs)-1:
            circle = plt.Circle((xs[i], ys[i]),
                                radius=cir_rad, edgecolor='pink',
                                facecolor='lightgreen', alpha=1.0)
        else:
            circle = plt.Circle((xs[i], ys[i]),
                                radius=cir_rad, edgecolor='red',
                                facecolor='yellow', alpha=1.0)
        ax.add_patch(circle)
        plt.annotate("{}".format(i), xy=(xs[i], ys[i]+3),
                        fontsize=10, ha="center", va="center")

    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    plt.savefig(file_name,bbox_inches='tight',pad_inches=0,dpi=300)
    #plt.show()
    plt.clf() 

def generate_gif(path, fps=1):

    images = []

    for file_name in sorted(os.listdir(path)):
        if file_name.endswith('.png'):
            file_path = os.path.join(path, file_name)
            images.append(imageio.imread(file_path))

    # Make it pause at the end so that the viewers can ponder
    #for _ in range(10):
    #    images.append(imageio.imread(file_path))

    imageio.mimsave(path+'/attention.gif', images, fps=fps)

def get_size_level(max_size, levels, area):

    interval = max_size/levels
    lvl = 0
    for i in range(levels):
        if area >= i*interval and area < (i+1)*interval:
            lvl = i+1
            break

    return lvl

# apply ior in a 3x3 area around a focal point
def ior_in_area(map, ym, xm, dims): 

    for i in range(3):
        if ym+i-1 < 0 or ym+i-1 >= dims[0]: continue
        for j in range(3):
            if xm+j-1 < 0 or xm+j-1 >= dims[1]: continue
            map[ym+i-1,xm+j-1] = 1

# get central detection coordinates in reference to the focal point
def get_local_coordinates(center, bbox):

    if len(center) != 2 or len(bbox) != 2:
        raise TypeError("Incorrect coordinate list dimensions!") 

    u = center[0]-bbox[0]
    v = center[1]-bbox[1]

    return np.array([u,v])

# covert from xywh to xyxy
def coordinate_converter(x, y, w, h):

    p1 = (int(x),int(y))
    p2 = (int(x+w),int(y+h))

    return p1,p2

# find bounding-box central coordinates
def box_center(p1,p2):

    center = [int((p1[0]+p2[0])/2),int((p1[1]+p2[1])/2)]

    return center

# check if bounding box belongs to the cell 
def in_cell(y_cell, x_cell, bbox, dims, h, w):

    int_x = w / dims[1]
    int_y = h / dims[0]
    x_min = int_x*x_cell
    x_max = int_x*(x_cell+1)
    y_min = int_y*y_cell
    y_max = int_y*(y_cell+1)

    box1 = [bbox[0],bbox[1],bbox[2],bbox[3]]
    box2 = [x_min,y_min,x_max,y_max]

    if(intersection_over_union(box1,box2) > 0.0):
        return True
    else:
        return False
    
# find cell center coordinates
def cell_center(y_cell, x_cell, dims, h, w):

    int_x = w / dims[1]
    int_y = h / dims[0]
    x_min = int_x*x_cell
    x_max = int_x*(x_cell+1)
    y_min = int_y*y_cell
    y_max = int_y*(y_cell+1)

    xc = (x_min + x_max)/2
    yc = (y_min + y_max)/2

    return (yc,xc)

# compute intersection over union 
def intersection_over_union(boxA, boxB):
        
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
        
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)    
	iou = interArea / float(boxAArea + boxBArea - interArea)
        
	return iou

# compute average of some metric in all cells of the map
def map_avg(map, dims):

    average = 0.0
    for i in range(dims[0]):
        for j in range(dims[1]):
            average += map[i][j]
    average = average / (dims[0]*dims[1])

    return average

def annotator(image,detections,labels,colors,n_classes):

    img = image.copy()
    font_scale = 1

    for det in detections:

        color = tuple([int(c) for c in colors[int(det[-1])]])
        xmin, ymin, xmax, ymax = det[:4]
    
        label_text = f"{labels[int(det[-1])]}: {max(det[4:(n_classes+4)]):.2f}"
        (text_width, text_height), baseline = cv.getTextSize(label_text,
                                                                cv.FONT_HERSHEY_SIMPLEX,
                                                                font_scale, 1)
    
        # Draw rectangle and label with the same color for the same class
        cv.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

    return img

# print map with metrics (for interpretation)
def print_map(map, dims):
    for i in range(dims[0]):
        for j in range(dims[1]):
            print("{:.3}".format(map[i][j]), end="\t")
        print("", end='\n')