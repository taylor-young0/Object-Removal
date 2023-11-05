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

    # Use list over a NumPy array because appending is faster
    # Can contain the same coordinates more than once!
    markup_locations = []
    
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
            # TODO: Update obj_removed_image 
            # Use markup locations to remove objects in the image
            # Create a mask based on the marked locations
            mask = np.zeros((height, width), dtype=np.uint8)
            for x, y in markup_locations:
                # Set marked locations in the mask to white
                mask[y, x] = 255

            # Use the mask to remove the marked objects from the image
            # Inpainting removes marked objects based on the generated mask
            obj_removed_image = cv2.inpaint(obj_removed_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            display_obj_removed_image(obj_removed_image)
        elif event == sg.WINDOW_CLOSED:
            break

    window.close()

def add_markup_locations(x, y, existing_locations):
    # update y because (0, 0) is bottom left of the image
    y = height - y

    delta = np.meshgrid([-2, -1, 0, 1, 2])

    dx, dy = np.meshgrid(delta, delta)
    cx, cy = np.full(dx.shape, x), np.full(dy.shape, y)

    surrounding_x = cx + dx
    surrounding_y = cy + dy

    surrounding_coordinates = np.stack((surrounding_x, surrounding_y), axis=-1)
    surrounding_coordinates_list = surrounding_coordinates.reshape(-1, 2).tolist()

    existing_locations.extend(surrounding_coordinates_list)

def markup_image(np_image, markup_locations, window):
    markups = np.array(markup_locations)
    np_image[markups[:, 1], markups[:, 0]] = 255
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