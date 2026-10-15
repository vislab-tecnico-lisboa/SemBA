# =============================================================================
# Deep Object Detection Model Functions
# =============================================================================

# =============================================================================
# 1. Inference dependencies and detector families
# =============================================================================

import torch
from utils.configs import *

# DETR
from transformers import DetrForObjectDetection, DetrImageProcessor

# D-FINE
from transformers import DFineForObjectDetection, AutoImageProcessor

# RT-DETR v1
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# RT-DETR v2
from transformers import RTDetrV2ForObjectDetection

# Custom model import
# from transformers import *


# =============================================================================
# 2. Model and preprocessing setup
# =============================================================================

def load_model(objdet, base_dim):
    """Load pretrained weights and a processor sized for square foveal layers.

    objdet selects a configured detector family; base_dim is the foveal crop
    side length in pixels. Returns (model, processor), loading checkpoint
    files through Transformers. The custom branch requires implementation
    of the model and processor constructors before it can be used.
    """

    if objdet == "detr":

        # DETR uses its own processor and includes a no-object logit.
        model = DetrForObjectDetection.from_pretrained(DETR_MODEL)
        processor = DetrImageProcessor.from_pretrained(DETR_MODEL)

    elif objdet == "dfine":

        # Resolve the processor from the D-FINE checkpoint configuration.
        model = DFineForObjectDetection.from_pretrained(DFINE_MODEL)
        processor = AutoImageProcessor.from_pretrained(DFINE_MODEL)

    elif objdet == "rtdetr":

        # Pair first-generation RT-DETR weights with their image processor.
        model = RTDetrForObjectDetection.from_pretrained(RTDETR_MODEL)
        processor = RTDetrImageProcessor.from_pretrained(RTDETR_MODEL)

    elif objdet == "rtdetr2":

        # RT-DETR v2 shares the RT-DETR image processor interface.
        model = RTDetrV2ForObjectDetection.from_pretrained(RTDETR2_MODEL)
        processor = RTDetrImageProcessor.from_pretrained(RTDETR2_MODEL)

    # This extension point requires both weights and model/processor wiring below.
    elif objdet == "custom":

        try:
            assert (
                CUSTOM_MODEL != None
            ), "Custom model is not defined. Please specify the pre-trained weights."
        except AssertionError as error:
            print(error)
            exit(-1)

        # Uncomment the following lines, replacing "CustomModel" with your model's name.
        # model = CustomModelForObjectDetection.from_pretrained(CUSTOM_MODEL)
        # processor = CustomModelProcessor.from_pretrained(CUSTOM_MODEL)

    else:

        raise NameError(
            f"The script is not ready to utilize the specified model: {objdet}"
        )

    # Match preprocessing dimensions to the foveator output.
    processor.size = {"shortest_edge": base_dim, "longest_edge": base_dim}

    return model, processor


# =============================================================================
# 3. Batched inference across foveal layers
# =============================================================================

def predict(layers, model, processor, score_thres=0.01, is_detr=False):
    """Detect all layers in one batch and return one detection list per layer.

    Each detection contains [xmin, ymin, xmax, ymax, *class_scores, class_id].
    Boxes use layer pixels; class scores follow the foreground category order.

    layers is a sequence of RGB image arrays in foveal order. model and
    processor come from load_model. score_thres filters the highest class
    probability before score renormalization; is_detr enables DETR-specific
    category handling. A layer with no retained predictions returns [].
    """

    # Remove unused COCO category slots from DETR to match compact class scores.
    deprecated_cats = [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]

    # Preserve layer order and disable gradient tracking for inference.
    inputs = processor(images=layers, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert normalized boxes to each layer's (height, width) pixel dimensions.
    target_sizes = torch.tensor(
        [layers[i].shape[:2] for i in range(len(layers))]
    )  # layer size (height, width)
    pred = post_process(
        outputs,
        target_sizes=target_sizes,
        conf_th=score_thres,
        deprecated=deprecated_cats,
        is_detr=is_detr,
    )

    return pred


# =============================================================================
# 4. Convert model outputs to the common SemBA detection format
# =============================================================================

def post_process(outputs, target_sizes, conf_th, deprecated, is_detr=True):
    """Filter predictions and convert boxes and scores to the shared format.

    DETR's final logit represents no object and is excluded from class scores.
    Its unused COCO slots are removed before renormalizing retained scores.

    outputs supplies logits and normalized center-format pred_boxes.
    target_sizes has shape (batch, 2), with each row storing (height, width).
    conf_th applies before unused category slots in deprecated are removed.
    Returns nested detection lists in batch order, using the layout documented
    in predict. Tensor-to-NumPy conversion expects CPU tensors without gradients.
    """

    # -------------------------------------------------------------------------
    # 4.1. Validate batch geometry
    # -------------------------------------------------------------------------

    out_logits, out_bbox = outputs.logits, outputs.pred_boxes

    assert len(out_logits) == len(
        target_sizes
    ), "Make sure that you pass in as many target sizes as the batch dimension of the logits"
    assert (
        target_sizes.shape[1] == 2
    ), "Each element of target_sizes must contain the size (h, w) of each image of the batch"

    # -------------------------------------------------------------------------
    # 4.2. Convert logits to class probabilities
    # -------------------------------------------------------------------------

    # Keep the full class distribution for later semantic belief fusion.
    prob = torch.softmax(out_logits, -1)
    if is_detr:
        _, labels = prob[..., :-1].max(-1)
        scores = prob[..., :-1]
    else:
        _, labels = prob[..., :].max(-1)
        scores = prob[..., :]

    # -------------------------------------------------------------------------
    # 4.3. Restore bounding boxes in layer pixels
    # -------------------------------------------------------------------------

    # Convert normalized (center_x, center_y, width, height) boxes to corners.
    cx, cy, w, h = out_bbox.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    # Scale x coordinates by width and y coordinates by height.
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    # -------------------------------------------------------------------------
    # 4.4. Filter queries and assemble detection rows
    # -------------------------------------------------------------------------

    # Each batch item yields its own list, including an empty list if filtered.
    results = []
    for i in range(scores.shape[0]):
        img = []
        probs = scores[i].numpy()

        # Apply the confidence threshold before removing any category slots.
        keep = probs.max(-1) > conf_th
        box = boxes[i].numpy()[keep]
        label = labels[i].numpy()[keep]
        probs = probs[keep]

        # Pack coordinates first, compact foreground scores next, and label last.
        for j in range(probs.shape[0]):
            det = []
            for k in range(box.shape[1]):
                det.append(box[j, k])
            for k in range(probs.shape[1]):
                if is_detr:
                    if k not in deprecated:
                        det.append(probs[j, k])
                else:
                    det.append(probs[j, k])

            # Normalize the retained foreground scores to sum to one.
            det[4:] = det[4:] / sum(det[4:])
            if is_detr:
                det.append(get_correct_cls_id(label[j], deprecated) - 1)
            else:
                det.append(label[j])
            img.append(det)
        results.append(img)

    return results


# =============================================================================
# 5. Translate sparse category indices
# =============================================================================

def get_correct_cls_id(cls, depre):
    """Shift a category ID past sorted unused slots, or return -1 for such a slot.

    cls is the original category index and depre is an ascending sequence
    of unused indices. The result subtracts the number of removed slots below
    cls; it does not apply any additional caller-specific label offset.
    """

    # Compare against the original ID while decrementing the returned index.
    ref = cls
    for i in range(len(depre)):
        if ref > depre[i]:
            cls = cls - 1
        elif ref == depre[i]:
            return -1
        else:
            break

    return cls
