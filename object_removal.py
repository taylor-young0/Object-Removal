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

    # Event loop
    while True:
        event, _ = window.read()

        if event == sg.WINDOW_CLOSED:
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

    global height, width
    height, width = image.shape[0], image.shape[1]

    display_image(image)

if __name__ == '__main__':
    main()