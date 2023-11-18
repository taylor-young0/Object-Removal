# Group project for CSCI 3240U
#
#   Aparnna Hariharan
#   Saawi Baloch
#   Taylor Young
#
# Parts of this code are based on code provided by Faisal Qureshi

import argparse
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

width = 0
height = 0
# Input image, never modified
base_image = np.array([])

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def remove_objects(image, marked_locations, radius=1):
    # Make a copy of orignial image
    result_image = [row[:] for row in image]

    for x, y in marked_locations:
        # Get the pixel values at the marked location
        pixel_value = image[y][x]

        for i in range(max(0, x - radius), min(len(image[0]), x + radius + 1)):
            for j in range(max(0, y - radius), min(len(image), y + radius + 1)):
                # Compare individual RGB components of the pixel values
                for c in range(3): # 0 = Red, 1 = Green, 2 = Blue
                    # If any RGB component of the pixel differs, replace the marked pixel
                    if image[j][i][c] != pixel_value[c]:
                        result_image[y][x] = image[j][i]
                        break  # Break if a non-marked pixel is found
    return result_image

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
            sg.Button("Remove Objects")
        ]
    ]

    # Create the window
    window = sg.Window('Object Removal', layout, finalize=True, size=(1000, 600))    
    window['-IMAGE-'].draw_image(data=image_data, location=(0, height))

    # Stores markup locations, 1 if that pixel is marked up, 0 otherwise
    markup_locations = np.zeros_like(base_image)
    
    # Base image + white pixel markup
    markedup_image = np.copy(base_image)
    # Base image + objects removed
    obj_removed_image = np.copy(base_image)

    # Event loop
    while True:
        event, values = window.read()

        if event == "-IMAGE-":
            x, y = values[event]
            add_markup_locations(x, y, existing_locations=markup_locations)
        elif event == "-IMAGE-+UP":
            markup_image(markedup_image, markup_locations, window)
        elif event == "Remove Objects":
            # Use markup locations to remove objects in the image
            markup_ys, markup_xs, _ = np.where(markup_locations == 1)
            markup_indexes = list(zip(markup_xs, markup_ys))
            obj_removed_image = remove_objects(base_image, markup_indexes)

            display_obj_removed_image(obj_removed_image)
        elif event == sg.WINDOW_CLOSED:
            break

    window.close()

def add_markup_locations(x, y, existing_locations):
    # Update y because (0, 0) is bottom left of the image
    y = height - y

    delta = np.array([-2, -1, 0, 1, 2])
    dx, dy = np.meshgrid(delta, delta)

    surrounding_x = np.clip(x + dx, 0, width - 1)
    surrounding_y = np.clip(y + dy, 0, height - 1)

    existing_locations[surrounding_y, surrounding_x] = 1

def markup_image(np_image, markup_locations, window):
    np_image[markup_locations == 1] = 255
    image_data = np_im_to_data(np_image)
    window['-IMAGE-'].erase()
    window['-IMAGE-'].draw_image(data=image_data, location=(0, height))

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

        if event == sg.WINDOW_CLOSED or "Close":
            break

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