"""Bounding-box helpers used by small UAV detection."""

from typing import Sequence

Box = Sequence[float]


def box_area(box: Box) -> float:
    """Return the area of an xyxy box. Degenerate boxes have area 0."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box_a: Box, box_b: Box) -> float:
    """Return IoU of two xyxy boxes in [0, 1]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = box_area(box_a) + box_area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union
