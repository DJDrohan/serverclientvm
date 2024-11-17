import cv2

def draw_text_with_border(image, text, position, font, font_scale, color, thickness):
    """
    Draws text on an image with a black border around it for better readability.

    Args:
        image (numpy.ndarray): The image to draw text on.
        text (str): The text to draw.
        position (tuple): (x, y) coordinates for the text position.
        font (int): OpenCV font type.
        font_scale (float): Font scale (size).
        color (tuple): BGR color for the main text.
        thickness (int): Thickness of the main text.

    Returns:
        None
    """
    x, y = position

    # Draw the black border by offsetting the text position in multiple directions
    border_color = (0, 0, 0)
    cv2.putText(image, text, (x - 1, y - 1), font, font_scale, border_color, thickness)
    cv2.putText(image, text, (x + 1, y - 1), font, font_scale, border_color, thickness)
    cv2.putText(image, text, (x - 1, y + 1), font, font_scale, border_color, thickness)
    cv2.putText(image, text, (x + 1, y + 1), font, font_scale, border_color, thickness)
    cv2.putText(image, text, (x, y - 1), font, font_scale, border_color, thickness)
    cv2.putText(image, text, (x, y + 1), font, font_scale, border_color, thickness)
    cv2.putText(image, text, (x - 1, y), font, font_scale, border_color, thickness)
    cv2.putText(image, text, (x + 1, y), font, font_scale, border_color, thickness)

    # Draw the main text in the specified color
    cv2.putText(image, text, position, font, font_scale, color, thickness)