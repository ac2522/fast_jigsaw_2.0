
import os
from argparse import ArgumentParser
from configparser import ConfigParser


import requests
import numpy as np
import cv2
import random
from random import uniform

DEFAULT_CONFIG = {
    "config": {
        "output_path": "jigsaw_pieces",
        "debug": True
    }
}

def load_image(path):
    """
    Load an image from a given path or URL.

    Parameters:
    path (str): File path or URL of the image to be loaded.

    Returns:
    numpy.ndarray: The loaded image.

    Raises:
    requests.HTTPError: If an HTTP error occurs while loading from a URL.
    FileNotFoundError: If the file path is invalid.
    IOError: If the image cannot be loaded.
    """
    try:
        if path.startswith(('http://', 'https://', 'www.')):
            response = requests.get(path)
            response.raise_for_status()
            img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        else:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File not found: {path}")
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        
        if img is None:
            raise IOError("Failed to load image.")
        
        return img
    except requests.HTTPError as e:
        raise IOError(f"HTTP error occurred: {e}")
    except Exception as e:
        raise IOError(f"Error loading image: {e}")


def handle_size_mismatch(img, pixels, size_mismatch):
    """
    Handle the size mismatch in an image based on the specified strategy.

    Parameters:
    - img (numpy.ndarray): The image to be processed.
    - pixels (int): The pixel dimensions to be matched.
    - size_mismatch (str): Strategy for handling size mismatch - 'crop', 'black', 'white', or 'stretch'.

    Returns:
    - numpy.ndarray: The processed image.

    Raises:
    - ValueError: If size mismatch handling is not specified correctly.
    """
    height, width, _ = img.shape
    width_mismatch = width % pixels
    height_mismatch = height % pixels

    if not width_mismatch and not height_mismatch:
        return img

    if size_mismatch is None:
        raise ValueError("Size mismatch handling strategy not specified.")

    if size_mismatch == "crop":
        return img[:height - height_mismatch, :width - width_mismatch]

    if size_mismatch in ["black", "white"]:
        new_width = width + (pixels - width_mismatch) % pixels
        new_height = height + (pixels - height_mismatch) % pixels
        new_img = np.full((new_height, new_width, 3), 255 if size_mismatch == "white" else 0, dtype=np.uint8)
        new_img[(new_height - height) // 2:(new_height + height) // 2,
                (new_width - width) // 2:(new_width + width) // 2] = img
        return new_img

    if size_mismatch == "stretch":
        new_width = width + pixels - width_mismatch
        new_height = height + pixels - height_mismatch
        return cv2.resize(img, (new_width, new_height))

    raise ValueError("Invalid size mismatch handling strategy. Choose from 'crop', 'black', 'white', or 'stretch'.")


def crop_image_around_lines(image, lines, overlap, rotate=False):
    """
    Isolates a region in an image based on provided lines. The area inside the lines is opaque, while the area outside is black and transparent.

    Parameters:
    - image (numpy.ndarray): Input image in BGR format.
    - lines (list of tuples): List of lines as [(x1, y1), (x2, y2), ...].
    - overlap (int): Overlap size for line areas.
    - rotate (bool): If True, randomly rotates the image by 0, 90, 180, or 270 degrees.

    Returns:
    - numpy.ndarray: Image with the region isolated.
    """
    alpha_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    alpha_minus = np.ones(image.shape[:2], dtype=np.uint8)

    size = image.shape[1]

    pts1 = np.concatenate([
        [[size, overlap - 1]],  # Top-right corner
        lines[0],   # First line endpoint
        lines[1],   # Second line endpoint
        [[overlap - 1, size]],  # Bottom-left corner
        [[size, size]]  # Bottom-right corner
    ])
    
    pts2 = np.concatenate([
        [[0, size]],  # Top-right corner
        [[0, size - overlap]],  # Top-right corner
        lines[2],   # First line endpoint
        lines[3],   # Second line endpoint
        [[size - overlap, 0]],  # Bottom-left corner
        [[size, 0]],  # Bottom-left corner
        [[size, size]]  # Bottom-right corner
    ])

    cv2.fillPoly(alpha_mask, [pts1], 255, lineType=cv2.LINE_4)
    cv2.fillPoly(alpha_minus, [pts2], 0, lineType=cv2.LINE_4)
    alpha_mask *= alpha_minus

    for channel in range(3):
        image[:, :, channel] = cv2.bitwise_and(image[:, :, channel], image[:, :, channel], mask=alpha_mask)
    image[:, :, 3] = alpha_mask

    if rotate:
        angle = random.choice([0, 1, 2, 3])

        if angle != 0:
            image = np.rot90(image, angle)

    return image


def validate_save_path(path):
    """
    Validates if the provided path is suitable for saving an image. Creates the directory if it does not exist.

    Parameters:
    - path (str): The file path to validate.

    Raises:
    - NotADirectoryError: If the specified path is not a directory.
    - PermissionError: If write permission is denied for the path.
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            raise PermissionError(f"Unable to create the directory: {path}. Error: {e}")

    if not os.path.isdir(path):
        raise NotADirectoryError(f"The specified path is not a directory: {path}")

    if not os.access(path, os.W_OK):
        raise PermissionError(f"Write permission denied for the path: {path}")


def init_matrices(size, overlap):
    """
    Initialize and manipulate matrices for further processing.

    Parameters:
    size (int): Scalar to resize the matrices.
    overlap (int): Scalar to adjust the overlap of matrices.

    Returns:
    tuple: A tuple containing manipulated matrices for various orientations.
    """
    h = ((np.array([
        [[0., 1.], [0.3, 1.1], [0.34, 1.05], [0.34, 1.05]],
        [[0.34, 1.05], [0.5, 1.], [0.4, 0.85], [0.4, 0.85]],
        [[0.4, 0.85], [0.3, 0.7], [0.5, 0.7], [0.5, 0.7]],
        [[0.5, 0.7], [0.7, 0.7], [0.6, 0.85], [0.6, 0.85]],
        [[0.6, 0.85], [0.5, 1.], [0.65, 1.05], [0.65, 1.05]],
        [[0.65, 1.05], [0.7, 1.1], [1., 1.], [1., 1.]]
    ]) * size) + overlap)
    v = h[::-1, ::-1, ::-1]

    horizontal = np.vstack([bezier_curve(point) for point in h])
    vertical = np.vstack([bezier_curve(point) for point in v])

    h_flip = flip_matrix(horizontal, size, 1)
    v_flip = flip_matrix(vertical, size, 0)

    
    return ([horizontal, h_flip], [vertical, v_flip],
            [reverse(horizontal.copy(), size, True), reverse(h_flip.copy(), size, True)],
            [reverse(vertical.copy(), size), reverse(v_flip.copy(), size)])



def flip_matrix(matrix, size, axis):
    """
    Flip a matrix horizontally or vertically.

    Parameters:
    matrix (numpy.ndarray): Input matrix to be flipped.
    size (int): Size parameter for flipping calculation.
    is_horizontal (bool): Flag to determine horizontal or vertical flip.

    Returns:
    numpy.ndarray: Flipped matrix.
    """
    flipped = matrix.copy()
    flipped[:, axis] = 2.7 * size - flipped[:, axis]
    return flipped

def reverse(array, size, is_horizontal=False):
    """
    Reverse the elements of an array with a specified shift.

    Parameters:
    array (numpy.ndarray): Array to be reversed.
    size (int): Scalar for shifting elements.
    is_horizontal (bool): Direction of the shift.

    Returns:
    numpy.ndarray: Reversed and shifted array.
    """
    array[:, int(is_horizontal)] -= size
    return array[::-1]

def bezier_curve(points):
    """
    Generate a Bezier curve from four control points.

    Parameters:
    points (numpy.ndarray): Control points for the Bezier curve.

    Returns:
    numpy.ndarray: Points constituting the Bezier curve.
    """
    t_values = np.linspace(0, 1, 100)
    one_minus_t = 1 - t_values
    one_minus_t_squared = one_minus_t**2
    one_minus_t_cubed = one_minus_t**3
    t_squared = t_values**2
    t_cubed = t_values**3

    x = one_minus_t_cubed * points[0][0] + 3 * one_minus_t_squared * t_values * points[1][0] + 3 * one_minus_t * t_squared * points[2][0] + t_cubed * points[3][0]
    y = one_minus_t_cubed * points[0][1] + 3 * one_minus_t_squared * t_values * points[1][1] + 3 * one_minus_t * t_squared * points[2][1] + t_cubed * points[3][1]

    return remove_duplicates(np.column_stack((x, y)).astype(int))

def remove_duplicates(arr):
    """
    Remove duplicate consecutive elements from an array.

    Parameters:
    arr (numpy.ndarray): Array from which duplicates are to be removed.

    Returns:
    numpy.ndarray: Array with duplicates removed.
    """
    return arr[np.append(np.any(np.diff(arr, axis=0), axis=1), True)]

def display_img(img):
    """
    Display an image using matplotlib.

    Parameters:
    img (numpy.ndarray): Image to be displayed.
    """
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def crop_image(image, top_left_x, top_left_y, crop_size, overlap):
    """
    Crops and optionally enlarges an image based on a specified area and overlap.

    Parameters:
    - image (numpy.ndarray): Input image.
    - top_left_x, top_left_y (int): Coordinates of the top-left corner for cropping.
    - crop_size (int): Size of the square crop area.
    - overlap (int): Size of the enlargement overlap on each side of the crop area.
    
    Returns:
    - numpy.ndarray: Cropped and possibly enlarged image. Returns an empty array if the crop area is outside the image bounds.
    """
    if overlap <= 0:
        return image[top_left_y:top_left_y+crop_size, top_left_x:top_left_x+crop_size]

    new_size = crop_size + 2 * overlap
    expanded_x, expanded_y = top_left_x - overlap, top_left_y - overlap

    if (expanded_x >= image.shape[1] or expanded_y >= image.shape[0] or
        expanded_x + new_size <= 0 or expanded_y + new_size <= 0):
        return np.zeros((new_size, new_size, image.shape[2]), dtype=image.dtype)

    canvas = np.zeros((new_size, new_size, image.shape[2]), dtype=image.dtype)
    boundary = lambda v, max_v: (max(v, 0), min(v + new_size, max_v))
    cx1, cx2 = boundary(expanded_x, image.shape[1])
    cy1, cy2 = boundary(expanded_y, image.shape[0])

    canvas_slice = slice(cy1 - expanded_y, cy2 - expanded_y), slice(cx1 - expanded_x, cx2 - expanded_x)
    image_slice = slice(cy1, cy2), slice(cx1, cx2)
    canvas[canvas_slice] = image[image_slice]

    return canvas

def create_custom_array_concise(width, height):
    """
    Creates a numpy array of specified dimensions with values from 0 to width*height - 1, randomly arranged with 0 as the first element.

    Parameters:
    - width, height (int): Dimensions of the array.

    Returns:
    - numpy.ndarray: Array with randomly arranged elements.
    """
    array = np.random.permutation(np.arange(1, width * height))
    return np.insert(array, 0, 0).reshape(width, height)

def remove_repeats(array):
    """
    Removes consecutive repeated elements from a numpy array.

    Parameters:
    - array (numpy.ndarray): Input array.

    Returns:
    - numpy.ndarray: Array with consecutive repeats removed.
    """
    diff_mask = np.any(array[1:] != array[:-1], axis=(1, 2))
    keep_mask = np.insert(diff_mask, 0, True)
    return array[keep_mask]



def modify_knob(size, overlap, horizontal=True):
    """
    Generates coordinates for a puzzle knob.

    Args:
        size (float): The scaling factor for the knob size.
        overlap (float): The overlapping factor for the knob.
        is_horizontal (bool): Flag to determine if the knob is horizontal or vertical.

    Returns:
        np.ndarray: An array of coordinates for the puzzle knob.
    """

    p1 = [uniform(.3, .35), uniform(1.01, 1.11)]
    p2 = [uniform(.38, .42), uniform(.84, .9)]
    depth = uniform(.66, .77)
    p4 = [uniform(.58, .62), uniform(.84, .9)]
    p5 = [uniform(.65, .7), uniform(1.01, 1.11)]


    x = uniform(0, 0.03)
    y = uniform(0, 0.03)

    p10 = [uniform(.16, p1[0] - 0.03), uniform(1.06, 1.15)]
    p40 = [uniform(p5[0] + 0.03, .84), uniform(1.06, 1.15)]

        
    h = (np.array([
        [[0., 1.], p10, [uniform(.25, min(p1[0] - 0.03, 2*p1[0]-0.12-p10[0])), p1[1] + x], p1],
        [p1, [uniform(.49, .57), 1.025 - x * 5], p2, p2],
        [p2, [uniform(.28, .37), depth + uniform(.02, .07)], [.48, depth], [.5, depth]],
        [[.5, depth], [.52, depth], [uniform(.63, .72), depth + uniform(.02, .07)], p4],
        [p4, [uniform(.43, .51), 1.025 - y * 5], p5, p5],
        [p5, [uniform(max(p5[0] + 0.03, 2*p5[0]-p40[0]+0.12), .75), p5[1] + y], p40, [1., 1.]]
    ]) * size) + overlap

    if not horizontal:
        h = h[::-1, ::-1, ::-1]

    h = np.vstack([bezier_curve(point) for point in h])

    if random.random() < 0.5:
        h[:, int(horizontal)] = 2.7 * size - h[:, int(horizontal)]
    return h



def validate_method(method):
    valid_methods = ["classic", "uniform", "random"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method: {method}. Expected one of {valid_methods}")


def main(image_path, piece_size, output_path="jigsaw_pieces",
         verbose=False, method="classic", size_mismatch="crop"):
    """
    Generates puzzle pieces from an image and saves them to the output path.

    Args:
        image_path (str): Path or URL to the source image.
        output_path (str): Path where the puzzle pieces will be saved.
        puzzle_piece_size (int): Size of each puzzle piece in piece_size.
        method (str): Shape generation method ('classic', 'uniform', or 'random').
                      Defaults to 'classic'.
    """
    validate_save_path(output_path)
    validate_method(method)

    uniform = method == "uniform"
    
    img = load_image(image_path)
    img = handle_size_mismatch(img, piece_size, size_mismatch)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # Add transparency
    
    # Cache shape values
    img_height, img_width = img.shape[:2]
    col_count = img_width // piece_size
    row_count = img_height // piece_size
    
    offset = int(0.35 * piece_size) + 1
    coords_hash = {}

    if uniform:
        horizontal, vertical, h_reversed, v_reversed = init_matrices(piece_size, offset)
        coords = np.random.randint(2, size=(col_count, row_count, 2))

    save_order = create_custom_array_concise(col_count, row_count)
    
    for col in range(col_count):
        for row in range(row_count):
            # crop to size
            cropped_img = crop_image(img, col * piece_size, row * piece_size, piece_size, offset)
            lines = []

            if row == 0: # top is edge
                lines.append([(offset+piece_size-1, offset), (offset, offset)])
            else:
                if uniform:
                    lines.append(h_reversed[coords[col, row - 1, 0]])
                else:
                    lines.append(reverse(coords_hash[(col, row - 1)][0], piece_size, True))

            if col == 0: # left side is edge
                lines.append([(offset, offset), (offset, offset + piece_size - 1)])
            else:
                if uniform:
                    lines.append(v_reversed[coords[(col - 1, row)][1]])
                else:
                    lines.append(reverse(coords_hash[(col - 1, row)][1], piece_size))

            if row + 1 == row_count: # bottom is edge
                lines.append([(offset, offset + piece_size), (offset + piece_size, offset + piece_size)])
                coords_hash[(col, row)] = [0]
            else: # horizontal
                if uniform or (method == "random" and random.random() < 0.5):
                    line = horizontal[coords[col, row, 0]]
                else:
                    line = modify_knob(piece_size, offset)

                if not uniform:
                    coords_hash[(col, row)] = [line]

                lines.append(line)

            if col + 1 == col_count: # right is edge
                lines.append([(offset + piece_size, offset + piece_size), (offset + piece_size, offset)])
            else:
                if uniform or (method == "random" and random.random() < 0.5):
                    line = vertical[coords[col, row, 1]]
                else:
                    line = modify_knob(piece_size, offset, horizontal=False)
                if not uniform:
                    coords_hash[(col, row)].append(line)
                lines.append(line)
            cropped_img = crop_image_around_lines(cropped_img, lines, offset, rotate= 0 < row+col)
            save_path = os.path.join(output_path, f"{save_order[col][row]}.png")

            cv2.imwrite(save_path, cropped_img)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('image_path', type=str, help='Path/URL to an input image')
    ap.add_argument('piece_size', type=int, help='Size (in pixels) of each puzzle piece')
    ap.add_argument('--output_path', type=str, default=None, help='Folder in which puzzle piece images should be saved')
    ap.add_argument('--verbose', '-v', action='store_true', help='Increase output verbosity')
    ap.add_argument('--config_file', '-c', default="./config/config.ini", action='store', 
                    help='Configuration ini file')
    ap.add_argument('--method', type=str, default="classic", help='The method with which the intersections are cut: uniform, classic, random')
    ap.add_argument('--size_mismatch', type=str, default="crop", help='If image size does not divide exactly by pieces: crop, stretch, black, white')
    args = ap.parse_args()

    cp = ConfigParser()
    cp.read_dict(DEFAULT_CONFIG)
    cp.read(args.config_file)

    output_path = args.output_path or cp.get("config", "output_path")
    verbose = cp.getboolean("config", "debug") or args.verbose
    main(args.image_path, args.piece_size, output_path, verbose, args.method, args.size_mismatch)

