from PIL import Image
import numpy as np
from scipy.stats import gaussian_kde

def image_to_board(image_name, image_folder='img/', scale=0.05):
    '''
    Takes an image and returns a matrix representing the image in ascii 
    characters.

        Params:
            image_name (string): The file name of the image.
            image_folder (string, optional): The folder of the image. Defaults 
                to 'img/'.
            scale (float, optional): The value to scale the image by. Defaults
                to 0.05.

        Returns:
            list of list of char: An ascii representation of the image based on 
                brightness of pixels.

        Raises:
            FileNotFoundException: If the file is not found in the image folder.
            PIL.UnidentifiedImageError: If the image cannot be opened and 
                identified.
    '''
    image = Image.open(image_folder + image_name)

    width, height = image.size
    width = int(width * scale)
    height = int(height * scale)
    image = image.resize((width, height))

    pixels = image.load()

    brightnesses = []

    for x in range(width):
        for y in range(height):
            total_brightness = sum(pixels[x, y])
            brightnesses.append(total_brightness)

    distribution = gaussian_kde(brightnesses)
    board = [[' ' for _ in range(width)] for _ in range(height)]

    for x in range(width):
        for y in range(height):
            percentile = distribution.integrate_box_1d(-np.inf, sum(pixels[x, y]))
            board[y][x] = brightness_percentile_to_ascii(percentile)

    return board

def brightness_percentile_to_ascii(percentile):
    '''
    Takes a percentile and converts it into its corresponding ascii char.

        Params:
            percentile (float): A float between 0 and 1 which represents a 
                pixel's percentile brightness. Values outside 0 and 1 are
                clamped.

        Returns:
            char: An ascii char.
    '''
    chars = [ ' ', '.', '\'', ',', ':', ';', '=', '+', '#', '@' ]

    if percentile > 1:
        percentile = 1
    if percentile < 0:
        percentile = 0

    return chars[int(percentile * 10)]

def list_to_string(l):
    '''
    Takes a 2d matrix and returns its string representation.

        Params:
            l (list of list): A 2d maxtix

        Returns:
            string: The concatenated values of the matrix separated only by new
                lines at each new row.
    '''
    s = ''
    for r in l:
        for a in r:
            s += str(a)
        s += '\n'
    return s

print(list_to_string(image_to_board('mona_lisa.jpg')))
