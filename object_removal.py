import argparse
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import os

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def resize_maintain_aspect_ratio(image, target_width, target_height):
    height, width, _ = image.shape
    aspect_ratio = width / height

    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

def save_image(image_to_save):
    filename = sg.popup_get_file('Save Image', save_as=True, file_types=(("PNG Files", "*.png"),))
    if filename:
        img = Image.fromarray(image_to_save)
        img.save(filename)

def draw_freely(canvas, event, values, drawing, canvas_height, image_before_location, image_before_location_end, markup_width, drawn_points):
    if event == sg.WIN_CLOSED:
        return
    if drawing:
        x, y = values['-IMAGE-']

        # Check if the coordinates are within the bounds of the original image
        if image_before_location[0] <= x <= image_before_location_end and 0 <= y <= canvas_height:
            point_size = values['markup_width']
            canvas.draw_point((x, y), size=point_size, color='white')
            drawn_points.append((x, y))

def draw_circle(canvas, event, values, drawing, canvas_height, image_before_location, image_before_location_end, markup_width, drawn_points):
    if event == sg.WIN_CLOSED:
        return
    if drawing:
        x, y = values['-IMAGE-']

        # Check if the coordinates are within the bounds of the original image
        if image_before_location[0] <= x <= image_before_location_end and 0 <= y <= canvas_height:
            point_size = values['markup_width']

            # If drawn_points is empty, initialize it with the current point
            if not drawn_points:
                drawn_points.append((x, y))
            else:
                # Draw the circle using the first and current points
                start_point = drawn_points[0]
                radius = int(np.sqrt((x - start_point[0]) ** 2 + (y - start_point[1]) ** 2))
                canvas.draw_circle(start_point, radius, fill_color='white', line_color='white', line_width=point_size)

def draw_rectangle(canvas, event, values, drawing, canvas_height, image_before_location, image_before_location_end, markup_width, drawn_points):
    if event == sg.WIN_CLOSED:
        return
    if drawing:
        x, y = values['-IMAGE-']

        # Check if the coordinates are within the bounds of the original image
        if image_before_location[0] <= x <= image_before_location_end and 0 <= y <= canvas_height:
            point_size = values['markup_width']

            # If drawn_points is empty, initialize it with the current point
            if not drawn_points:
                drawn_points.append((x, y))
            else:
                # Draw the rectangle using the first and current points
                start_point = drawn_points[0]
                top_left = (min(start_point[0], x), min(start_point[1], y))
                bottom_right = (max(start_point[0], x), max(start_point[1], y))

                canvas.draw_rectangle(top_left, bottom_right, fill_color='white', line_color='white', line_width=point_size)

def remove_circle(np_image, drawn_points, radius=10):
    copy_img = np_image.copy()
    for point in drawn_points:
        x, y = point

        # Ensure indices are within the valid range
        y = max(0, min(y, copy_img.shape[0] - 1))
        x = max(0, min(x, copy_img.shape[1] - 1))

        # Fill the circle region with the original image values
        cv2.circle(copy_img, (x, y), radius, copy_img[y, x].tolist(), -1)

    return copy_img

def remove_rectangle(np_image, drawn_points, radius=5):
    copy_img = np_image.copy()
    
    for point in drawn_points:
        x, y = point

        # Ensure indices are within the valid range
        y = max(0, min(y, copy_img.shape[0] - 1))
        x = max(0, min(x, copy_img.shape[1] - 1))

        # Get the pixel values at the marked location
        pixel_value = copy_img[y, x]

        for i in range(max(0, x - radius), min(copy_img.shape[1], x + radius + 1)):
            for j in range(max(0, y - radius), min(copy_img.shape[0], y + radius + 1)):
                # Calculate Euclidean distance from the center of the circle
                distance = np.sqrt((x - i) ** 2 + (y - j) ** 2)

                # If the pixel is within the circle, replace its value
                if distance <= radius:
                    copy_img[j, i] = pixel_value

    return copy_img


def remove_objects(np_image, drawn_points, radius=5):
    copy_img = np_image.copy()
    for point in drawn_points:
        x, y = point

        # Ensure indices are within the valid range
        y = max(0, min(y, copy_img.shape[0] - 1))
        x = max(0, min(x, copy_img.shape[1] - 1))

        # Get the pixel values at the marked location
        pixel_value = copy_img[y, x]

        for i in range(max(0, x - radius), min(copy_img.shape[1], x + radius + 1)):
            for j in range(max(0, y - radius), min(copy_img.shape[0], y + radius + 1)):
                # Compare individual RGB components of the pixel values
                for c in range(3):  # 0 = Red, 1 = Green, 2 = Blue
                    # If any RGB component of the pixel differs, replace the marked pixel
                    if copy_img[j, i, c] != pixel_value[c]:
                        copy_img[j, i] = pixel_value
                        break  # Break if a non-marked pixel is found
    return copy_img

def display_image(np_image_before, np_image_after, image_filename):
    filename = os.path.basename(image_filename)

    # Convert numpy arrays to data that sg.Graph can understand
    image_data_before = np_im_to_data(np_image_before)
    image_data_after = np_im_to_data(np_image_after)

    height = 300

    # Get the dimensions of the original and resized images
    orig_height, orig_width, _ = np_image_before.shape
    new_height, new_width, _ = np_image_after.shape

    # Calculate the canvas width for each image
    canvas_width_before = int(height * (orig_width / orig_height))
    canvas_width_after = int(height * (new_width / new_height))

    # Define the layout with the adjusted canvas widths for each image
    layout = [
        [sg.Graph(
            canvas_size=(canvas_width_before + canvas_width_after + 100, height),
            graph_bottom_left=(0, 0),
            graph_top_right=(canvas_width_before + canvas_width_after, height),
            key='-IMAGE-',
            change_submits=True,
            drag_submits=True
        )],
        [sg.Text(f'File: {filename}', size=(40, 1))],  # Display the filename
        [sg.Text(f'Size: {orig_width} x {orig_height} pixels', size=(40, 1))],  # Display the size
        [
            sg.Column([
                [sg.Button('Save'), sg.Button('Reset')],
                [sg.Button('Exit')]
            ], justification='left'),
            sg.Column([
                [sg.Button('Draw freely'), sg.Button('Circle'), sg.Button('Rectangle')],
                [sg.Button('Fill Enclosures'), sg.Button('Remove Objects')],
                [sg.Text('Markup width (px):'), sg.Slider(range=(1, 21), default_value=5, orientation='h', key='markup_width')]
            ], justification='left', pad=((50, 0), (40, 0)))
        ]
    ]

    # Create the window
    window = sg.Window('Display Image', layout, finalize=True)

    # Display the before and after images side by side
    window['-IMAGE-'].draw_image(data=image_data_before, location=(0, height))
    window['-IMAGE-'].draw_image(data=image_data_after, location=(canvas_width_before, height))

    drawing = False
    freeDraw_clicked=False
    circle_clicked=False
    rectangle_clicked=False

    drawn_points = []  # Initialize an empty list to store drawn points

    # Event loop
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        elif event == 'Save':
            save_image(np_image_after)
        elif event == 'Reset':
            # Reset both before and after images
            window['-IMAGE-'].draw_image(data=np_im_to_data(np_image_before), location=(0, height))
            window['-IMAGE-'].draw_image(data=np_im_to_data(np_image_before), location=(canvas_width_before, height))
            np_image_after = np_image_before.copy()  # Reset the modified image to the original state
            drawing = False
            freeDraw_clicked=False
            circle_clicked=False
            rectangle_clicked=False
            drawn_points = []  # Clear the drawn points when resetting

        elif event == 'Draw freely':
            drawing = not drawing
            freeDraw_clicked=True

        elif event == 'Circle':
            drawing = True  # Set drawing to True when Circle button is clicked
            circle_clicked=True

        elif event == 'Rectangle':
            drawing = True
            rectangle_clicked=True
            drawn_points = []  # Reset drawn points for the rectangle
            
        elif event == 'Remove Objects':
            if(freeDraw_clicked==True):
                copy_img = np_image_after
                obj_removed_image = remove_objects(copy_img, drawn_points)  # Use a copy of the drawn points
                window['-IMAGE-'].draw_image(data=np_im_to_data(obj_removed_image), location=(canvas_width_before, height))  # Draw the modified image
                drawn_points = []  # Clear the drawn points after removing objects
                np_image_after = obj_removed_image
            
            elif(circle_clicked==True):
                obj_removed_image = remove_circle(np_image_after, drawn_points)  # Use a copy of the drawn points
                window['-IMAGE-'].draw_image(data=np_im_to_data(obj_removed_image), location=(canvas_width_before, height))  # Draw the modified image
                drawn_points = []  # Clear the drawn points after removing objects
                np_image_after = obj_removed_image
                circle_clicked=False
            
            elif(rectangle_clicked==True):
                obj_removed_image = remove_rectangle(np_image_after, drawn_points)
                window['-IMAGE-'].draw_image(data=np_im_to_data(obj_removed_image), location=(canvas_width_before, height))
                drawn_points = []  # Clear the drawn points after removing objects
                rectangle_clicked = False

        elif drawing and event == '-IMAGE-':
            if circle_clicked:
                draw_circle(window['-IMAGE-'], event, values, drawing, height, (0, height), canvas_width_before - 50, values['markup_width'], drawn_points)
            elif rectangle_clicked:
                draw_rectangle(window['-IMAGE-'], event, values, drawing, height, (0, height), canvas_width_before - 50, values['markup_width'], drawn_points)
            else:
             draw_freely(window['-IMAGE-'], event, values, drawing, height, (0, height), canvas_width_before-50, values['markup_width'], drawn_points)
   
    window.close()

def main():
    parser = argparse.ArgumentParser(description='A simple image viewer.')
    parser.add_argument('file', action='store', help='Image file.')
    args = parser.parse_args()

    print(f'Loading {args.file} ... ', end='')
    image = cv2.imread(args.file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'{image.shape}')

    aspect_ratio = image.shape[1] / image.shape[0]

    print(f'Resizing the image to fit within the canvas while maintaining aspect ratio ...', end='')
    # Calculate the canvas width based on the image dimensions and the desired height
    canvas_width = int(300 * aspect_ratio)
    image = resize_maintain_aspect_ratio(image, canvas_width, 300)
    print(f'{image.shape}')

    # Create a copy of the original image for display
    image_copy = image.copy()
    display_image(image_copy, image, args.file)  # Pass the filename to the display_image function

if __name__ == '__main__':
    main()
