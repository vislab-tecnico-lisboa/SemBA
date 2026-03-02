#############################################################################################################################
# Sematic Information Model Functions
#############################################################################################################################

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

def load_model(objdet, base_dim):

    if objdet == "detr":

        # DETR definition
        model = DetrForObjectDetection.from_pretrained(DETR_MODEL)
        processor = DetrImageProcessor.from_pretrained(DETR_MODEL)

    elif objdet == "dfine":

        model = DFineForObjectDetection.from_pretrained(DFINE_MODEL)
        processor = AutoImageProcessor.from_pretrained(DFINE_MODEL)

    elif objdet == "rtdetr":

        # RT-DETR definition
        model = RTDetrForObjectDetection.from_pretrained(RTDETR_MODEL)
        processor = RTDetrImageProcessor.from_pretrained(RTDETR_MODEL)

    elif objdet == "rtdetr2":

        # RT-DETRv2 definition
        model = RTDetrV2ForObjectDetection.from_pretrained(RTDETR2_MODEL)
        processor = RTDetrImageProcessor.from_pretrained(RTDETR2_MODEL)

    else:

        raise NameError(f"The script is not ready to utilize the selected model: {objdet}")

    # adapt the porcessor size to the actual size of the layers
    processor.size = {"shortest_edge": base_dim, "longest_edge": base_dim}

    return model, processor

def predict(layers, model, processor, score_thres=0.01, is_detr=False):

    # COCO 2017 deprecated categories - only used by DETR
    deprecated_cats = [0,12,26,29,30,45,66,68,69,71,83,91]

    inputs = processor(images=layers, return_tensors="pt")
    with torch.no_grad(): outputs = model(**inputs)
    target_sizes = torch.tensor([layers[i].shape[:2] for i in range(len(layers))]) # layer size (height, width)
    pred = post_process(outputs,
                        target_sizes=target_sizes,
                        conf_th=score_thres,
                        deprecated=deprecated_cats,
                        is_detr=is_detr)

    return pred

def post_process(outputs, target_sizes, conf_th, deprecated, is_detr=True):
       
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        assert len(out_logits) == len(
            target_sizes
        ), "Make sure that you pass in as many target sizes as the batch dimension of the logits"
        assert (
            target_sizes.shape[1] == 2
        ), "Each element of target_sizes must contain the size (h, w) of each image of the batch"

        prob = torch.softmax(out_logits, -1)
        if is_detr: 
            _, labels = prob[..., :-1].max(-1)
            scores = prob[..., :-1]
        else: 
            _, labels = prob[..., :].max(-1)
            scores = prob[..., :]

        # convert to [x0, y0, x1, y1] format
        cx, cy, w, h = out_bbox.unbind(-1)
        x_min = cx - 0.5 * w
        y_min = cy - 0.5 * h
        x_max = cx + 0.5 * w
        y_max = cy + 0.5 * h
        boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for i in range(scores.shape[0]):
            img = []
            probs = scores[i].numpy()
            keep = probs.max(-1) > conf_th
            box = boxes[i].numpy()[keep]
            label = labels[i].numpy()[keep]
            probs = probs[keep]    
            for j in range(probs.shape[0]):
                det = []
                for k in range(box.shape[1]):
                    det.append(box[j,k])
                for k in range(probs.shape[1]):
                    if is_detr:
                        if k not in deprecated:
                            det.append(probs[j,k])
                    else: det.append(probs[j,k])
                det[4:] = det[4:]/sum(det[4:])
                if is_detr:
                    det.append(get_correct_cls_id(label[j], deprecated)-1)
                else:
                    det.append(label[j])
                img.append(det)
            results.append(img)

        return results

def get_correct_cls_id(cls, depre):

    ref = cls
    for i in range(len(depre)):
        if ref > depre[i]:
            cls = cls - 1
        elif ref == depre[i]:
            return -1
        else: break

    return cls