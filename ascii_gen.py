from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

plt.switch_backend('Agg')

def image_to_ascii(image_name, image_folder='img/', n_components=1, scale=0.05):
    '''
    Takes an image and returns a matrix representing the image in ascii 
    characters.

        Params:
            image_name (string): The file name of the image.
            image_folder (string, optional): The folder of the image. Defaults 
                to 'img/'.
            n_components (int, optional): The number of gaussian distributions
                to fit to. 
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
    p = image_to_percentile(image_name, image_folder=image_folder, scale=scale)
    return [list(map(brightness_percentile_to_ascii, row)) for row in p]

def image_to_percentile(image_name, image_folder='img/', n_components=1, scale=0.05):
    '''
    Takes an image and returns a matrix representing the brightness of each
    pixel as a percent compared to a gaussian distribution.

        Params:
            image_name (string): The file name of the image.
            image_folder (string, optional): The folder of the image. Defaults 
                to 'img/'.
            n_components (int, optional): The number of gaussian distributions
                to fit to. 
            scale (float, optional): The value to scale the image by. Defaults
                to 0.05.

        Returns:
            list of list of float: A representation of the relative brightness
                of each pixel.

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

    brightnesses = np.zeros((height, width))
    board = np.zeros((height, width))

    pixels = image.load()

    for y in range(height):
        for x in range(width):
            brightnesses[y][x] = sum(pixels[x, y])

    gmm = GaussianMixture(n_components=n_components)
    X = brightnesses.flatten().reshape(-1,1)
    gmm.fit(X)
    x = np.linspace(-5, 10, 192)
    x = x.reshape(-1, 1)

    plt.hist(X, bins=30, density=True, alpha=0.5)

    for i in range(len(gmm.weights_)):
        y = gmm.weights_[i] * stats.norm.pdf(x, gmm.means_[i], np.sqrt(gmm.covariances_[i]))
        plt.plot(x, y)

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Histogram of Sampled Data with Gaussian Mixture Model')
    plt.savefig('a.png')

    for y in range(height):
        for x in range(width):
            for i in range(n_components):
                board[y][x] = gmm.weights_[i] * stats.norm.cdf(brightnesses[y, x], gmm.means_[i,0], np.sqrt(gmm.covariances_[i,0,0]))

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

print(list_to_string(image_to_ascii('mona_lisa.jpg', n_components=2, scale=0.03)))
