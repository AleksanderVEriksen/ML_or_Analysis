import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def rediger_bilde_med_slidere(filename):
    img = cv2.imread(filename)
    start_kernel = 31
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3))

    ax1.set_title('Original')
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ax2.set_title('Gråskala')
    ax2.imshow(img_gray, cmap='gray')

    ax3.set_title('Uskarp')
    img_blur = cv2.GaussianBlur(img, (start_kernel, start_kernel), 0)
    ax3.imshow(cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB))

    ax4.set_title('Kanter')
    img_edges = cv2.Canny(img_gray, start_kernel, start_kernel)
    ax4.imshow(cv2.cvtColor(img_edges, cv2.COLOR_BGR2RGB))

    blur_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    blur_slider = Slider(blur_ax, 'Kjernestørrelse', 1, 100, valinit=start_kernel, valstep=2)

    def update(val):
        ksize = int(blur_slider.val)
        if ksize % 2 == 0:
            ksize += 1
        img_blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
        img_edges = cv2.Canny(img_gray, ksize, ksize)
        ax3.imshow(cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB))
        ax4.imshow(cv2.cvtColor(img_edges, cv2.COLOR_BGR2RGB))
        fig.canvas.draw_idle()

    blur_slider.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

    def reset(event):
        blur_slider.reset()

    button.on_clicked(reset)

    plt.show()

if __name__ == '__main__':
    filename = 'C:\\Users\\eriks\\OneDrive\\Random_koder\\Bilder\\suesser.jpg'
    rediger_bilde_med_slidere(filename)