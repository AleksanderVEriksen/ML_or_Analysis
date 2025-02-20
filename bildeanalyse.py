import cv2
import unittest
from PIL import Image
import numpy as np
import array

filvei = 'C:\\Users\eriks\\OneDrive\\Random_koder\\Bilder\\suesser.jpg'  # Endre til ditt bilde

def vis_bilde(tittel, bilde):
    """Viser et bilde med gitt tittel."""
    cv2.imshow(tittel, bilde)
    cv2.waitKey(0)  # Venter på tastetrykk før lukker vinduet
    cv2.destroyAllWindows()

def gråskaler_bilde(bilde):
    """Gjør bildet til gråskala ved hjelp av RGB til gråskala konvertering."""
    return cv2.cvtColor(bilde, cv2.COLOR_BGR2GRAY)

def gaussisk_uskarpt_bilde(bilde):
    """Lagrer et gaussisk uskarpt bilde. Bruker en kernelsize på 5x5 med standardavvik på 2."""
    image = bilde
    man_gaus = np.ones((5,5), dtype=float)/25
    
    img = cv2.filter2D(src=image, ddepth=-1, kernel=man_gaus)

    """for i in range(1,image.shape[0]-1):
            for j in range(1,image.shape[1]-1):
                dx = (image[i-1][j-1] + image[i-1][j] + image[i-1][j+1])
                dy = (image[i][j-1] - 4*image[i][j] + image[i][j+1])
                dz = (image[i+1][j-1] +  image[i][j] + image[i+1][j+1])
                man_gaus[i][j] = dx + dy + dz"""
    return img

def laplacian_bilde(image):
    """Gjør et laplacian-bilde, som lar seg bruke til å finne kantinformasjon."""
    image = gråskaler_bilde(image)
    man_lap = np.zeros(image.shape, dtype=float)
    for i in range(1,image.shape[0]-1):
            for j in range(1,image.shape[1]-1):
                dx = (image[i,j+1] - 2*image[i,j] + image[i,j-1])
                dy = (image[i+1,j] - 2*image[i,j] + image[i-1,j])
                man_lap[i,j] = dx + dy
    return man_lap 


def hovedfunksjon(filvei):
    """Hovedfunksjon som laster et bilde, viser originalt og endret bilder."""
    # Laste bilde fra fil
    original_bilde = cv2.imread(filvei)
    if original_bilde is None:
        print(f"Kunne ikke laste bilde fra fil: {filvei}")
        return

    # Vis originalt bilde
    vis_bilde("Original bilde", original_bilde)

    # Lagre og vis gråskalert bilde
    gråskaler = gråskaler_bilde(original_bilde)
    vis_bilde("Gråskala bilde", gråskaler)

    # Lagre og vis gaussisk uskarpt bilde
    gaussisk = gaussisk_uskarpt_bilde(original_bilde)
    vis_bilde("Gaussisk uskarpt bilde", gaussisk)

    # Lagre og vis laplacian-bilde
    laplacian = laplacian_bilde(gråskaler)
    vis_bilde("Laplacian bilde", laplacian)



class Testclass(unittest.TestCase):

    def test_bilde(self):
        bilde = cv2.imread(filvei)
        bilde2 = np.asarray(Image.open(filvei))[:,:,::-1]
        
        np.testing.assert_allclose(bilde, bilde2, atol=1)
    
    def test_grå(self):
        bilde = cv2.imread(filvei)
        grå = gråskaler_bilde(bilde)
        
        np.testing.assert_array_equal(len(grå.shape), 2)
        np.testing.assert_array_almost_equal(grå, 255)
        np.testing.assert_array_less(grå, 0)
        


    def test_laplacian_filter(self):
        image = cv2.imread(filvei)
        man_lap = laplacian_bilde(image)
        lap = cv2.Laplacian(gråskaler_bilde(image), cv2.CV_64F)

        for x in range(len(lap)):
            for y in range(len(lap[x])):
                self.assertAlmostEqual(abs(man_lap[x][y]), abs(lap[x][y]), delta=1)


    def test_gaussian_blur_filter(self):
        image = cv2.imread(filvei)
        gaus = cv2.GaussianBlur(image, (5, 5), 2)
        
        man_gaus = gaussisk_uskarpt_bilde(image)

        for x in range(len(gaus)):
            for y in range(len(gaus[x])):
                for z in range(len(gaus[x][y])):
                    self.assertAlmostEqual(abs(man_gaus[x][y][z]), abs(gaus[x][y][z]), delta=1)
        
if __name__ == "__main__":
    unittest.main()
    # Test reasoning used