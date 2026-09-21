from uav_detection.bbox import box_area, box_iou


def test_box_area_of_unit_square():
    assert box_area((0, 0, 1, 1)) == 1.0


def test_box_area_of_invalid_box_is_zero():
    assert box_area((2, 2, 1, 1)) == 0.0


def test_iou_of_identical_boxes_is_one():
    box = (10, 20, 30, 40)
    assert box_iou(box, box) == 1.0


def test_iou_of_non_overlapping_boxes_is_zero():
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_of_half_overlap():
    iou = box_iou((0, 0, 2, 2), (1, 0, 3, 2))
    assert abs(iou - 1 / 3) < 1e-9
