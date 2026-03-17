import os
import re

import numpy as np

def slice_with_overlap(lst, n, k):
    if n <= 0 or k < 0:
        raise ValueError("n must be greater than 0 and k must be non-negative")
    result = []
    i = 0
    while i < len(lst):
        result.append(lst[i:i + n])
        i += max(1, n - k)  # Ensure progress even if k >= n
    return result


def sort_images_by_number(image_paths):
    def extract_number(path):
        filename = os.path.basename(path)
        # Match decimal or integer number in filename
        match = re.search(r'\d+(?:\.\d+)?', filename)
        return float(match.group()) if match else float('inf')

    return sorted(image_paths, key=extract_number)

def downsample_images(image_names, downsample_factor):
    """
    Downsamples a list of image names by keeping every `downsample_factor`-th image.
    
    Args:
        image_names (list of str): List of image filenames.
        downsample_factor (int): Factor to downsample the list. E.g., 2 keeps every other image.

    Returns:
        list of str: Downsampled list of image filenames.
    """
    return image_names[::downsample_factor]


def apply_point_visualization(
    colors,
    dynamic_mask=None,
    low_conf_mask=None,
    vis_uncertainty="red",
    vis_low_conf="transparent",
):
    """
    Apply visualization styles to point colors and return which points should be hidden.

    Dynamic uncertainty points take priority over low-confidence points when both masks overlap.
    """
    colors_out = np.array(colors, copy=True)
    if colors_out.size == 0:
        return colors_out, np.empty((0,), dtype=bool)

    num_points = colors_out.shape[0]
    if dynamic_mask is None:
        dynamic_mask = np.zeros((num_points,), dtype=bool)
    else:
        dynamic_mask = np.asarray(dynamic_mask).reshape(-1).astype(bool)
    if low_conf_mask is None:
        low_conf_mask = np.zeros((num_points,), dtype=bool)
    else:
        low_conf_mask = np.asarray(low_conf_mask).reshape(-1).astype(bool)

    if dynamic_mask.shape[0] != num_points:
        raise ValueError("Dynamic mask length must match point count.")
    if low_conf_mask.shape[0] != num_points:
        raise ValueError("Low-confidence mask length must match point count.")

    low_conf_only = low_conf_mask & ~dynamic_mask
    max_value = 1.0 if colors_out.max() <= 1.0 else 255.0
    red = np.array([max_value, 0.0, 0.0], dtype=colors_out.dtype)
    white = np.array([max_value, max_value, max_value], dtype=colors_out.dtype)

    if vis_low_conf == "red":
        colors_out[low_conf_only] = red
    elif vis_low_conf == "white":
        colors_out[low_conf_only] = white

    if vis_uncertainty == "red":
        colors_out[dynamic_mask] = red
    elif vis_uncertainty == "white":
        colors_out[dynamic_mask] = white

    hidden_mask = np.zeros((num_points,), dtype=bool)
    if vis_low_conf == "transparent":
        hidden_mask |= low_conf_only
    if vis_uncertainty == "transparent":
        hidden_mask |= dynamic_mask

    return colors_out, hidden_mask
