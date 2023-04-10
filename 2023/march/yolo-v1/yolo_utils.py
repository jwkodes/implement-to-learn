from typing import List, Tuple, Union
import numpy as np
import random
import cv2

try:
    # Python >3.8
    from typing import Literal
except:
    # Python <=3.7
    from typing_extensions import Literal


def generate_tiles(
    image_height: int,
    image_width: int,
    target_res_h: int,
    target_res_w: int,
    box_format: Literal["corner", "center"] = "corner",
    perturb_boxes: int = None,
) -> List[Tuple[int, int, int, int]]:
    # Count number of full tiles and number of partial tiles
    num_full_tiles_x = image_width // target_res_w
    num_full_tiles_y = image_height // target_res_h
    num_partial_tiles_x = 1 if (image_width % target_res_w) > 0 else 0
    num_partial_tiles_y = 1 if (image_height % target_res_h) > 0 else 0

    # Generate crop_bboxes (x1,y1,x2,y2), in corner format, for each tile
    crop_bboxes = []
    # Full tiles
    for i in range(num_full_tiles_x):
        for j in range(num_full_tiles_y):
            bbox = (
                i * target_res_w,
                j * target_res_h,
                (i + 1) * target_res_w,
                (j + 1) * target_res_w,
            )
            crop_bboxes.append(bbox)
    # Half tiles
    for i in range(num_full_tiles_x):
        for j in range(num_partial_tiles_y):
            bbox = (
                i * target_res_w,
                max(0, image_height - target_res_h),
                (i + 1) * target_res_w,
                image_height,
            )
            crop_bboxes.append(bbox)
    for i in range(num_partial_tiles_x):
        for j in range(num_full_tiles_y):
            bbox = (
                max(0, image_width - target_res_w),
                j * target_res_h,
                image_width,
                (j + 1) * target_res_w,
            )
            crop_bboxes.append(bbox)
    if num_full_tiles_x > 0 and num_full_tiles_y > 0:
        crop_bboxes.append(
            (
                max(0, image_width - target_res_w),
                max(0, image_height - target_res_h),
                image_width,
                image_height,
            )
        )

    # Randomly perturb boxes by a small amount
    if perturb_boxes is not None and perturb_boxes > 0:
        for i in range(len(crop_bboxes)):
            temp_x1, temp_y1, temp_x2, temp_y2 = crop_bboxes[i]
            temp_x1 += random.randint(-perturb_boxes, perturb_boxes)
            temp_y1 += random.randint(-perturb_boxes, perturb_boxes)
            temp_x2 += random.randint(-perturb_boxes, perturb_boxes)
            temp_y2 += random.randint(-perturb_boxes, perturb_boxes)

            # Ensure that box does not go out of image boundary
            temp_x1 = max(0, temp_x1)
            temp_y1 = max(0, temp_y1)
            temp_x2 = min(image_width, temp_x2)
            temp_y2 = min(image_height, temp_y2)

            crop_bboxes[i] = (temp_x1, temp_y1, temp_x2, temp_y2)

    # Convert box format if necessary
    if box_format == "center":
        crop_bboxes = bbox_conversion(crop_bboxes, "corner2center")

    return crop_bboxes


def read_yolo_annotation_file(
    annotation_filename: str,
) -> List[Tuple[int, float, float, float, float]]:
    with open(annotation_filename, "r") as f:
        lines = f.readlines()

    new_lines = []
    for i, line in enumerate(lines):
        line_parts = line.strip().split()
        class_idx, x_center, y_center, width, height = map(float, line_parts)
        class_idx = int(class_idx)
        new_lines.append((class_idx, x_center, y_center, width, height))

    return new_lines


def write_yolo_annotation_file(
    annotation_filename: str, annotations: List[Tuple[int, float, float, float, float]]
) -> None:
    # Convert data to str
    new_lines = []
    for line in annotations:
        class_idx, x_center, y_center, width, height = line
        str_line = (
            f"{int(class_idx)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        )
        new_lines.append(str_line)

    # Write to file
    with open(annotation_filename, "w") as f:
        f.writelines(new_lines)


def crop_annotations(
    image: np.ndarray,
    original_annotations: List[Tuple[int, float, float, float, float]],
    crop_bbox: Tuple[int, int, int, int],
    keep_truncated=False,
    max_trucation_ratio=0.5,
) -> List[Tuple[int, float, float, float, float]]:

    """crop_annotations _summary_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    # Original image dimensions
    og_img_height = image.shape[0]
    og_img_width = image.shape[1]

    # New image dimensions
    new_img_height = crop_bbox[3] - crop_bbox[1]
    new_img_width = crop_bbox[2] - crop_bbox[0]

    if crop_bbox[0] < 0 or crop_bbox[1] < 0:
        raise ValueError("Invalid crop box. Value should be >= 0.")
    if crop_bbox[2] > og_img_width or crop_bbox[3] > og_img_height:
        raise ValueError(
            "Invalid crop box. Value should not be greater than the original image dimensions."
        )
    if new_img_width <= 0 or new_img_height <= 0:
        raise ValueError("Crop box is negative in width or height.")

    # Adjust the YOLO annotation for the cropped image
    new_annotations = []
    for i, line in enumerate(original_annotations):
        class_idx, x_center, y_center, width, height = line

        # Un-normalise
        x_center_pix, y_center_pix, width_pix, height_pix = (
            x_center * og_img_width,
            y_center * og_img_height,
            width * og_img_width,
            height * og_img_height,
        )
        x_center_pix -= crop_bbox[0]
        y_center_pix -= crop_bbox[1]

        # Normalise based on new dimensions
        x_center = x_center_pix / new_img_width
        y_center = y_center_pix / new_img_height
        width = width_pix / new_img_width
        height = height_pix / new_img_height

        if width == 0.0 or height == 0.0 or width * height == 0.0:
            continue

        if not keep_truncated:
            # Ignore if the annotation is outside of the cropped region
            if (
                x_center - width / 2 < 0
                or y_center - height / 2 < 0
                or x_center + width / 2 > 1.0
                or y_center + height / 2 > 1.0
            ):
                continue
        elif keep_truncated:
            # Calculate trucation ratio
            full_obj_area = width * height
            new_tlhc_x = max(0, x_center - width / 2)
            new_tlhc_x = min(1.0, new_tlhc_x)
            new_brhc_x = min(1.0, x_center + width / 2)
            new_brhc_x = max(0.0, new_brhc_x)
            new_tlhc_y = max(0.0, y_center - height / 2)
            new_tlhc_y = min(1.0, new_tlhc_y)
            new_brhc_y = min(1.0, y_center + height / 2)
            new_brhc_y = max(0.0, new_brhc_y)
            new_height = new_brhc_y - new_tlhc_y
            new_width = new_brhc_x - new_tlhc_x
            visible_obj_area = new_height * new_width

            if (
                visible_obj_area / full_obj_area < (1.0 - max_trucation_ratio)
                or visible_obj_area == 0.0
            ):
                # Ignore this label as there's too much truncation
                continue
            else:
                new_x_center = 0.5 * (new_tlhc_x + new_brhc_x)
                new_y_center = 0.5 * (new_tlhc_y + new_brhc_y)

                x_center = new_x_center
                y_center = new_y_center
                width = new_width
                height = new_height

        new_annotations.append((int(class_idx), x_center, y_center, width, height))

    return new_annotations


def bbox_conversion(
    bboxes: List[
        Union[Tuple[float, float, float, float], Tuple[int, float, float, float, float]]
    ],
    conversion: Literal["corner2center", "center2corner"],
) -> List[
    Union[Tuple[float, float, float, float], Tuple[int, float, float, float, float]]
]:
    out_bboxes = []
    for box in bboxes:
        if conversion == "corner2center":
            if len(box) == 4:
                # Bounding box only, no class
                x1, y1, x2, y2 = box
            elif len(box) == 5:
                # Bounding box and class
                cls, x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            if len(box) == 4:
                out_bboxes.append((x_center, y_center, w, h))
            elif len(box) == 5:
                out_bboxes.append((cls, x_center, y_center, w, h))
        elif conversion == "center2corner":
            if len(box) == 4:
                # Bounding box only, no class
                x_center, y_center, w, h = box
            elif len(box) == 5:
                # Bounding box and class
                cls, x_center, y_center, w, h = box
            x1 = x_center - w / 2.0
            y1 = y_center - h / 2.0
            x2 = x1 + w
            y2 = y1 + h
            if len(box) == 4:
                out_bboxes.append((x1, y1, x2, y2))
            elif len(box) == 5:
                out_bboxes.append((cls, x1, y1, x2, y2))
        else:
            raise ValueError("Invalid conversion method requested.")
    return out_bboxes


def get_axis_aligned_bbox(
    vertices: Tuple[float, float, float, float, float, float, float, float]
) -> Tuple[float, float, float, float]:
    """get_axis_aligned_bbox _summary_

    Args:
        vertices of a non-axis aligned box (list): [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        list: vertices of an axis aligned box [xmin, ymin, xmax, ymax]
    """
    # Split the list of vertices into x-coordinates and y-coordinates
    x_coords = vertices[::2]
    y_coords = vertices[1::2]

    # Compute the minimum and maximum x-coordinates and y-coordinates
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # Construct tuple of vertices for the 4-sided polygon
    bbox_vertices = (min_x, min_y, max_x, max_y)

    return bbox_vertices


# def convert_bbox_to_yolo(
#     image_width: int, image_height: int, box_vertices: Tuple[float, float, float, float]
# ) -> Tuple[float, float, float, float]:
#     x_min, y_min, x_max, y_max = box_vertices
#     box_width = x_max - x_min
#     box_height = y_max - y_min
#     x_center = (x_min + x_max) / 2
#     y_center = (y_min + y_max) / 2
#     yolo_x = x_center / image_width
#     yolo_y = y_center / image_height
#     yolo_width = box_width / image_width
#     yolo_height = box_height / image_height
#     return (yolo_x, yolo_y, yolo_width, yolo_height)


def unnormalise_yolo_bbox(
    image_width: int,
    image_height: int,
    annotations: List[
        Union[Tuple[float, float, float, float], Tuple[int, float, float, float, float]]
    ],
):
    new_lines = []
    for line in annotations:
        if len(line) == 4:
            # Bounding box only, no class
            x_center_norm, y_center_norm, w_norm, h_norm = line
            x_center = x_center_norm * image_width
            y_center = y_center_norm * image_height
            w = w_norm * image_width
            h = h_norm * image_height
            new_lines.append((x_center, y_center, w, h))
        elif len(line) == 5:
            # Bounding box and class
            cls, x_center_norm, y_center_norm, w_norm, h_norm = line
            x_center = x_center_norm * image_width
            y_center = y_center_norm * image_height
            w = w_norm * image_width
            h = h_norm * image_height
            new_lines.append((int(cls), int(x_center), int(y_center), int(w), int(h)))
    return new_lines


def normalise_bbox_yolo(
    image_width: int,
    image_height: int,
    annotations: List[
        Union[Tuple[float, float, float, float], Tuple[int, float, float, float, float]]
    ],
):

    new_lines = []
    for line in annotations:
        if len(line) == 4:
            # Bounding box only, no class
            x_center, y_center, w, h = line
            x_center_norm = x_center / image_width
            y_center_norm = y_center / image_height
            w_norm = w / image_width
            h_norm = h / image_height
            new_lines.append((x_center_norm, y_center_norm, w_norm, h_norm))
        elif len(line) == 5:
            # Bounding box and class
            cls, x_center, y_center, w, h = line
            x_center_norm = x_center / image_width
            y_center_norm = y_center / image_height
            w_norm = w / image_width
            h_norm = h / image_height
            new_lines.append((cls, x_center_norm, y_center_norm, w_norm, h_norm))
    return new_lines


def overlay_annotation(
    image: np.ndarray,
    annotations: List[
        Union[Tuple[float, float, float, float], Tuple[int, float, float, float, float]]
    ],
    normalised: bool = False,
    box_format: Literal["corner", "center"] = "center",
) -> None:
    image_height, image_width = image.shape[0], image.shape[1]

    # Change box format to corner if it is center
    if box_format == "center":
        annotations = bbox_conversion(annotations, "center2corner")
    elif box_format not in ["corner", "center"]:
        raise ValueError("Invalid value for box_format.")

    # Unnormalise bounding box info if it is normalised
    if normalised:
        annotations = unnormalise_yolo_bbox(image_width, image_height, annotations)

    # Overlay annotation on image
    for line in annotations:
        if len(line) == 4:
            # Bounding box only, no class
            x1, y1, x2, y2 = line
            cls = -1
        elif len(line) == 5:
            # Bounding box and class
            cls, x1, y1, x2, y2 = line
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Define text to display
        text = str(cls)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(text, font, 1, 1)
        # Calculate position of text above rectangle
        text_x = x1
        text_y = int(y1 - text_size[1] / 2)

        # Put text above rectangle
        cv2.putText(image, text, (text_x, text_y), font, 1, (0, 255, 0), 2)
