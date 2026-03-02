#################################################################################################
# SemBA Configurations File                                                
#################################################################################################

# Main Constant Definition
MAX_FIX = 6 # maximum amount of fixations until search/exploration is terminated 
X_CELLS, Y_CELLS = 32, 20 # size of the semantic cell grid map
TERMINATION_THRESH = 0.1 # termination cell confidence threshold (between 0 and 1)
CONF_THRESH = 0.01 # minimum confidence threshold to discard an object detection

# Deep Object Detectors (with Pre-Trained Weights)
DETECTORS = {'detr', 'dfine', 'rtdetr', 'rtdetr2'}
DETR_MODEL = "facebook/detr-resnet-50"
DFINE_MODEL = "ustc-community/dfine_x_coco"
RTDETR_MODEL = "PekingU/rtdetr_r50vd"
RTDETR2_MODEL = "PekingU/rtdetr_v2_r18vd"

# Target Object Categories
CLASS_NAMES = [ 'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
                'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']