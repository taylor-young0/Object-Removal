# Group project for CSCI 3240U
#
#   Aparnna Hariharan
#   Saawi Baloch
#   Taylor Young
#
# Parts of this code are based on code provided by Faisal Qureshi

import argparse
import PySimpleGUI as sg
from PIL import Image, ImageDraw
from io import BytesIO
from scipy import ndimage
import numpy as np
import cv2

width = 0
height = 0
# Input image, never modified
base_image = np.array([])
# mouse drag continuity
last_x, last_y = None, None

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def remove_objects(image, marked_locations, removal_radius=1):
    # Make a copy of the original image
    modified_image = [row[:] for row in image]

    # Loop through each marked location
    for marked_x, marked_y in marked_locations:
        # Get the color of the marked pixel
        marked_pixel_color = image[marked_y][marked_x]

        # Range of x values within the specified radius
        x_range = range(max(0, marked_x - removal_radius), min(len(image[0]), marked_x + removal_radius + 1))

        # Range of y values within the specified radius
        y_range = range(max(0, marked_y - removal_radius), min(len(image), marked_y + removal_radius + 1))

        # Check nearby pixels within the specified radius
        for current_x in x_range:
            for current_y in y_range:
                # Compare color channels of the marked pixel with nearby pixel
                for color_channel in range(3):  # 0 = Red, 1 = Green, 2 = Blue
                    # When any of the color channel differs, replace the marked pixel
                    if image[current_y][current_x][color_channel] != marked_pixel_color[color_channel]:
                        modified_image[marked_y][marked_x] = image[current_y][current_x]
                        break  # If a non-marked pixel is found

    return modified_image

def display_image(np_image):
    # Convert numpy array to data that sg.Graph can understand
    image_data = np_im_to_data(np_image)

    # Define the layout
    graph_image_column = [
        [
            sg.Graph(
                canvas_size=(width, height),
                graph_bottom_left=(0, 0),
                graph_top_right=(width, height),
                key='-IMAGE-',
                background_color='white',
                change_submits=True,
                drag_submits=True
            ),
        ]
    ]
    
    scrollable_graph_image_column = [
        [
            sg.Column(graph_image_column, scrollable=True, size=(900, 500))
        ]
    ]

    layout = [
        scrollable_graph_image_column,
        [
            sg.Button("Reset"),
            sg.Button("Fill Enclosures"),
            sg.Button("Remove Objects"),
            sg.Text(size=(70, 0)),
            sg.Text("Markup width (px)"),
            sg.Slider((1, 21), resolution=2, default_value=5, orientation='horizontal', key="markup_width")
        ]
    ]

    # Create the window
    window = sg.Window('Object Removal', layout, finalize=True, size=(1000, 600))    
    window['-IMAGE-'].draw_image(data=image_data, location=(0, height))

    # Stores markup locations, 1 if that pixel is marked up, 0 otherwise
    markup_locations = np.zeros_like(base_image)[:,:,0]
    
    # Base image + white pixel markup
    markedup_image = np.copy(base_image)
    # Base image + objects removed
    obj_removed_image = np.copy(base_image)
 
    global last_x, last_y

    # Event loop
    while True:
        event, values = window.read()

        if event == "-IMAGE-":
            x, y = values[event]
            markup_width = values["markup_width"]
            add_markup_locations(x, y, markup_width, existing_locations=markup_locations)
        elif event == "-IMAGE-+UP":
            last_x, last_y = None, None
            markup_image(markedup_image, markup_locations, window)
        elif event == "Reset":
            markup_locations = np.zeros_like(base_image)[:,:,0]
            markedup_image = np.copy(base_image)
            last_x, last_y = None, None
            reset_markup_image(markedup_image, window)
        elif event == "Fill Enclosures":
            markup_locations = ndimage.binary_fill_holes(markup_locations)
            markup_image(markedup_image, markup_locations, window)
        elif event == "Remove Objects":
            # Use markup locations to remove objects in the image
            markup_ys, markup_xs = np.where(markup_locations == 1)
            markup_indexes = list(zip(markup_xs, markup_ys))
            obj_removed_image = remove_objects(np.copy(base_image), markup_indexes)
            display_obj_removed_image(obj_removed_image)
        elif event == sg.WINDOW_CLOSED:
            break

    window.close()

def add_markup_locations(x, y, markup_width, existing_locations):
    global last_x, last_y

    # Update y because (0, 0) is bottom left of the image
    y = height - y

    # Convert to PIL Image for drawing
    image = Image.fromarray(base_image)
    draw = ImageDraw.Draw(image)

    # Interpolate points between last and current mouse positions using draw.line
    if last_x is not None and last_y is not None:
        line_coordinates = [(last_x, last_y), (x, y)]
        draw.line(line_coordinates, fill=(255, 0, 0), width=int(markup_width))

        # Update existing_locations based on the drawn line
        drawn_image = np.array(image)
        drawn_line_mask = np.all(drawn_image == [255, 0, 0], axis=-1)
        existing_locations[drawn_line_mask] = 1

    # Save the last point for drag continuity
    last_x, last_y = x, y

def markup_image(np_image, markup_locations, window):
    np_image[markup_locations == 1] = 255
    image_data = np_im_to_data(np_image)
    window['-IMAGE-'].erase()
    window['-IMAGE-'].draw_image(data=image_data, location=(0, height))

def reset_markup_image(markedup_image, window):
    image_data = np_im_to_data(markedup_image)
    window['-IMAGE-'].erase()
    window['-IMAGE-'].draw_image(data=image_data, location=(0, height))

def save_image(np_image):
    filename = sg.popup_get_file('Save Image', default_extension=".png", save_as=True, file_types=(("PNG Files", "*.png"), ("All Files", "*.*")))

    if filename:
        # Convert numpy array to data that sg.Graph can understand
        image_data = np_im_to_data(np_image)

        # Save the image to the specified file
        with open(filename, 'wb') as f:
            f.write(image_data)


def display_obj_removed_image(np_image):
    # Convert numpy array to data that sg.Graph can understand
    image_data = np_im_to_data(np_image)

    # Define the layout
    graph_image_column = [
        [
            sg.Graph(
                canvas_size=(width, height),
                graph_bottom_left=(0, 0),
                graph_top_right=(width, height),
                key='-IMAGE-',
                background_color='white',
                change_submits=True,
                drag_submits=True
            ),
        ]
    ]

    scrollable_graph_image_column = [
        [
            sg.Column(graph_image_column, scrollable=True, size=(900, 500))
        ]
    ]

    layout = [
        scrollable_graph_image_column,
        [
            sg.Button("Close"),
            sg.Button("Save")
        ]
    ]

    # Create the window
    window = sg.Window('Object Removal - Result', layout, finalize=True, size=(1000, 600))    
    window['-IMAGE-'].draw_image(data=image_data, location=(0, height))

    # Event loop
    while True:
        event, _ = window.read()

        if event == sg.WINDOW_CLOSED or event == "Close":
            break
        elif event == "Save":
            save_image(np_image)

    window.close()

def main():
    parser = argparse.ArgumentParser(description='A simple object remover.')

    parser.add_argument('file', action='store', help='Image file.')
    args = parser.parse_args()

    filename = args.file

    print(f'Loading {filename} ... ', end='')
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'{image.shape}')

    global base_image, height, width
    base_image = image
    height, width = base_image.shape[0], base_image.shape[1]

    display_image(image)

if __name__ == '__main__':
    main()
