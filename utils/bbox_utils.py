def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def get_bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    return x2 - x1
