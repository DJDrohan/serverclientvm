import cv2
import numpy as np

def resize_and_pad(img, target_width=640, target_height=480, background_color=(255, 255, 255), interpolation=cv2.INTER_LINEAR):
    """
    Resizes the image to fit within the target dimensions while maintaining aspect ratio.
    Pads the resized image with a specified background color to meet the exact target size.

    Arguments:
        img = source image
        target_width = target width in pixels
        target_height = target height in pixels
        background_color = background color set to white
        interpolation: Interpolation method for resizing, default is cv2.INTER_LINEAR.

    Returns:
        numpy.ndarray: Resized and padded image with the specified target dimensions.
    """
    if img is None or not isinstance(img, np.ndarray): #makes sure a numpy image array was sent
        raise ValueError("Input must be a valid numpy ndarray image.")

    # Calculate the scaling factors for width and height
    scale_width = target_width / img.shape[1]
    scale_height = target_height / img.shape[0]
    scale = min(scale_width, scale_height)  # Use the smaller scale factor to maintain aspect ratio

    # Calculate new dimensions
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)

    # Resize the image
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=interpolation)

    # Create a blank canvas with the target dimensions and background color
    canvas = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)

    # Calculate top-left corner to center the image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place the resized image on the canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized

    return canvas