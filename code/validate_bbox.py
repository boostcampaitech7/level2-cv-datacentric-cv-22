from detect import get_bboxes
import numpy as np


def extract_true_bboxes(gt_score_map, gt_geo_map, score_thresh=0.5, nms_thresh=0.2):
    return get_bboxes(gt_score_map, gt_geo_map, score_thresh=score_thresh, nms_thresh=nms_thresh)


def ensure_bbox_format(bboxes):
    # Check if bboxes has the shape (n, 9) and convert to (n, 4, 2)
    if bboxes.shape[1] == 9:
        # Assuming that the first 8 elements are the coordinates, reshape accordingly
        bboxes = bboxes[:, :8].reshape(-1, 4, 2)
    elif bboxes.shape[1] == 8:  # Already in compatible shape for (n, 4, 2)
        bboxes = bboxes.reshape(-1, 4, 2)
    elif bboxes.shape[1] == 4:
        # Handle rect format (xmin, ymin, xmax, ymax)
        bboxes = np.array([[
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]]
        ] for bbox in bboxes])
    else:
        raise ValueError(f"Unexpected bbox format: {bboxes.shape}")

    return bboxes

