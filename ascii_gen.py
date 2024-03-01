from PIL import Image
import numpy as np
from scipy.stats import gaussian_kde

def image_to_board(image_name, image_folder='img/', scale=0.05):
    image = Image.open(image_folder + image_name)
    width, height = image.size
    width = int(width * scale)
    height = int(height * scale)
    print(width, height)
    image = image.resize((width, height))
    pixels = image.load()
    average_total_brightness = 0
    brightnesses = []
    for x in range(width):
        for y in range(height):
            total_brightness = sum(pixels[x, y])
            average_total_brightness += total_brightness
            brightnesses.append(total_brightness)
    distribution = gaussian_kde(brightnesses)
    average_total_brightness //= width * height
    board = [[' ' for _ in range(width)] for _ in range(height)]
    for x in range(width):
        for y in range(height):
            percentile = distribution.integrate_box_1d(-np.inf, sum(pixels[x, y]))
            board[y][x] = brightness_percentile_to_ascii(percentile)
            '''
            if sum(pixels[x, y]) > average_total_brightness:
                board[y][x] = '.'
            else:
                board[y][x] = '#'
            '''
    print(average_total_brightness)
    return board

def brightness_percentile_to_ascii(percentile):
    chars = [ ' ', '.', '\'', ',', ':', ';', '=', '+', '#', '@' ]
    return chars[int(percentile * 10)]

def list_to_string(l):
    s = ''
    for r in l:
        for a in r:
            s += a
        s += '\n'
    return s

print(list_to_string(image_to_board('mona_lisa.jpg')))
