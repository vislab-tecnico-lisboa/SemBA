# =============================================================================
# SemBA Configurations File
# =============================================================================

# =============================================================================
# 1. Search limits and confidence thresholds
# =============================================================================

# Search budget and grid resolution. Grid arrays have shape (Y_CELLS, X_CELLS).
MAX_FIX = 6  # subsequent fixations after the initial center fixation
X_CELLS, Y_CELLS = 32, 20  # size of the semantic cell grid map

# Search success and detector filtering use different probability thresholds.
TERMINATION_THRESH = 0.1  # target probability required at the current fixation cell
CONF_THRESH = 0.01  # retain detections whose highest class score exceeds this value

# =============================================================================
# 2. Detector choices and pretrained model identifiers
# =============================================================================

# Identifiers are passed to the matching Transformers model and image processor.
DETECTORS = {"detr", "dfine", "rtdetr", "rtdetr2", "custom"}
DETR_MODEL = "facebook/detr-resnet-50"
DFINE_MODEL = "ustc-community/dfine_x_coco"
RTDETR_MODEL = "PekingU/rtdetr_r50vd"
RTDETR2_MODEL = "PekingU/rtdetr_v2_r18vd"

# Custom weights also require model/processor constructors in detectors.py.
CUSTOM_MODEL = None  # replace with pre-trained weights

# =============================================================================
# 3. Shared foreground vocabulary
# =============================================================================

# Keep this order aligned with the compact score vectors in detectors.py.
# Foreground category order shared by detector scores and belief vectors.
# Background occupies index 0 here only; target class IDs are therefore one-based.
CLASS_NAMES = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
