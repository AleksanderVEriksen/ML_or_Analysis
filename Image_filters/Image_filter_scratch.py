import numpy as np
import cv2
import unittest

# Hardkodet sti til bildet
Image_path = "C:\\Users\eriks\\OneDrive\\Random_koder\\Bilder\\suesser.jpg"  # Endre denne til stien til ditt bilde

# Funksjon for å konvertere til gråskala
def to_grayscale(image):
    """
    Påfører et gråskalert filter til et gitt bilde.
    
    Args:
        image (np.ndarray): Inndata bilde som skal behandles.
    
    Returns:
        np.ndarray: Bildet etter behandlig med gråskalering.
    """
    # Gjennomsnittlig vekt for RGB-kanalene
    weight = np.array([0.299, 0.587, 0.114])
    
    # Beregner gråskala ved å bruke vektet gjennomsnitt
    image = image[:,:,::-1]
    grayscale_image = np.dot(image, weight)
    
    # Konverterer til heltall
    grayscale_image = grayscale_image.astype(np.uint8)
    
    return grayscale_image
import numpy as np
from scipy.signal import convolve2d

def gaussian_blur(image, kernel_size=5, sigma=1):
    """
    Påfører et Gaussian blur til et gitt bilde ved bruk av en konvolusjonskernel.
    
    Args:
        image (np.ndarray): Inndata bilde som skal gjøres uskarpt.
        kernel_size (int): Størrelsen på Gaussian-blur kernel. Må være et oddetall. Standard er 5.
        sigma (float): Standardavvik for Gaussian-distribusjonen. Standard er 1.
    
    Returns:
        np.ndarray: Uskarpet bilde.
    """
    # Kernel-størrelse må være en oddetall
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Genererer Gaussian kernel
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    y = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    x, y = np.meshgrid(x, y)
    gaussian_kernel = np.exp(-((x**2 + y**2) / (2 * sigma**2)))
    gaussian_kernel /= np.sum(gaussian_kernel)
    
    # Bruker scipy's convolve2d for å utføre konvolusjon
    return convolve2d(image, gaussian_kernel, mode='same', boundary='symm').astype(np.uint8)

def laplacian_filter(image):
    """
    Påfører et Laplacian filter til et gitt bilde for å fremheve kantene.
    
    Args:
        image (np.ndarray): Inndata bilde som skal behandles.
    
    Returns:
        np.ndarray: Bildet etter behandlig med Laplacian filter.
    """
    # Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    
    # Bruker scipy's convolve2d for å utføre konvolusjon
    return convolve2d(image, laplacian_kernel, mode='same', boundary='symm').astype(np.uint8)
def main():
    # Les inn bildet
    image = cv2.imread(Image_path)
    # Hvis bildet ikke blir lastet inn, sjekk stien og prøv igjen
    if image is None:
        raise FileNotFoundError(f"Bildet ble ikke funnet på stien {Image_path}")
    # Konverter til gråskala
    grayscale_image = to_grayscale(image)
    # Gaussian blur
    gaussian_blurred_image = gaussian_blur(grayscale_image)
    # Laplacian filter
    laplacian_image = laplacian_filter(grayscale_image)

    # Vis resultatene
    cv2.imshow('Original', image)
    cv2.imshow('Gråskala', grayscale_image)
    cv2.imshow('Gaussian Blur', gaussian_blurred_image)
    cv2.imshow('Laplacian Filter', laplacian_image)

    # Vent på tastetrykk og lukk vinduer
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Testcase(unittest.TestCase):

    def test_image_read(self):
        import os
        Image = cv2.imread(Image_path) if os.path.exists(Image_path) else None
        self.assertIsNotNone(Image)
    
    def test_grayscale_shape(self):
        Image = cv2.imread(Image_path)
        gray_image = to_grayscale(Image)
        self.assertLess(len(gray_image.shape), len(Image.shape))

    
    def test_grayscale_pixle_values(self):
        Image = cv2.imread(Image_path)
        gray_image = to_grayscale(Image)

        for x in range(len(gray_image)):
            Flag = any((y>255 or y<0) for y in gray_image[x])
            self.assertFalse(Flag)


if __name__ == "__main__":
    #main()
    unittest.main()