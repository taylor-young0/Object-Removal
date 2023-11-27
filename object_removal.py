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
from skimage.draw import line_aa

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

def remove_objects(image, marked_locations, radius=1):
    # Make a copy of orignial image
    result_image = [row[:] for row in image]

    for i in range(8):
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
        drawn_line_mask = np.all(drawn_image == [255, 0, 0], axis=-1)  # Assuming red color for the drawn line
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