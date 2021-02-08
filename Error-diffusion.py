import numpy as np
import matplotlib.pyplot as plt
import imageio


def halftone_function(I, box_size=16):

    try:
        I = np.mean(I, axis=2)
    except IndexError:
        pass

    halftone_image = np.zeros((I.shape[0], I.shape[1]))

    number_of_rows = I.shape[0] // box_size
    number_of_cols = I.shape[1] // box_size

    for row in range(0, number_of_rows):
        for col in range(0, number_of_cols):

            Isub = I[row * box_size: (row + 1) * box_size,
                   col * box_size: (col + 1) * box_size]

            mean = np.mean(Isub)

            halftone_matrix_sub = np.ones((box_size, box_size)) * 255

            mean = 255 - mean

            if mean == 0:
                pass
            elif mean == 255:
                halftone_matrix_sub = np.zeros((box_size, box_size))
            else:
                for x in range(box_size):
                    for y in range(box_size):
                        if ((x - box_size/2) ** 2 + (y - box_size/2) ** 2) *\
                                (255/box_size**2) <= mean / np.pi:

                            halftone_matrix_sub[x][y] = 0

            halftone_image[row * box_size: (row + 1) * box_size,
                    col * box_size: (col + 1) * box_size] = halftone_matrix_sub

    return halftone_image


def FS_bw(I, f=1):

    try:
        I = np.mean(I, axis=2)
    except IndexError:
        pass

    x_boundary = I.shape[0]
    y_boundary = I.shape[1]

    for y in range(y_boundary):
        for x in range(x_boundary):


            pixel = I[x, y]


            new_pixel = int(abs(round(f * pixel / 255) * (255 / f)))

            I[x, y] = new_pixel

            errorPix = pixel - new_pixel

            if x < x_boundary - 1:
                fs_pixel = I[x + 1, y] + round(errorPix * (7 / 16))

                I[x + 1, y] = fs_pixel

            if x > 1 and y < y_boundary - 1:
                fs_pixel = I[x - 1, y + 1] + round(errorPix * 3 / 16)

                I[x - 1, y + 1] = fs_pixel

            if y < y_boundary - 1:
                fs_pixel = I[x, y + 1] + round(errorPix * 5 / 16)

                I[x, y + 1] = fs_pixel

            if x < x_boundary - 1 and y < y_boundary - 1:
                fs_pixel = I[x + 1, y + 1] + round(errorPix * 1 / 16)

                I[x + 1, y + 1] = fs_pixel


    return I



def floyd_steinberg_color(I, f=1):
    x_boundary = I.shape[0]
    y_boundary = I.shape[1]
    counter = 0
    for y in range(1, y_boundary):
        for x in range(1, x_boundary):

            oldred, oldgreen, oldblue = I[x, y]

            newred = int(abs(round(f * oldred/255) * (255/f)))
            newgreen = int(abs(round(f * oldgreen / 255) * (255 / f)))
            newblue = int(abs(round(f * oldblue / 255) * (255 / f)))


            I[x, y] = newred, newgreen, newblue

            errorRed = oldred - newred

            errorGreen = oldgreen - newgreen
            errorBlue = oldblue - newblue


            if x < x_boundary - 1:
                red = I[x + 1, y][0] + round(errorRed * (7 / 16))
                green = I[x + 1, y][1] + round(errorGreen * (7 / 16))
                blue = I[x + 1, y][2] + round(errorBlue * (7 / 16))

                I[x + 1, y] = red, green, blue

            if x > 1 and y < y_boundary - 1:
                red = I[x - 1, y + 1][0] + round(errorRed * 3 / 16)
                green = I[x - 1, y + 1][1] + round(errorGreen * 3 / 16)
                blue = I[x - 1, y + 1][2] + round(errorBlue * 3 / 16)

                I[x - 1, y + 1] = red, green, blue

            if y < y_boundary - 1:
                red = I[x, y + 1][0] + round(errorRed * 5 / 16)
                green = I[x, y + 1][1] + round(errorGreen * 5 / 16)
                blue = I[x, y + 1][2] + round(errorBlue * 5 / 16)

                I[x, y + 1] = red, green, blue

            if x < x_boundary - 1 and y < y_boundary - 1:
                red = I[x + 1, y + 1][0] + round(errorRed * 1 / 16)
                green = I[x + 1, y + 1][1] + round(errorGreen * 1 / 16)
                blue = I[x + 1, y + 1][2] + round(errorBlue * 1 / 16)

                I[x + 1, y + 1] = red, green, blue
    print(counter)

    return I


def floyd_steinberg(I, f=1):

    try:
        I = np.mean(I, axis=2)
    except IndexError:
        pass

    x_boundary = I.shape[0]
    y_boundary = I.shape[1]

    for y in range(0, y_boundary-1):
        for x in range(1, x_boundary-1):

            pixel = I[x, y]

            new_pixel = int(abs(round(f * pixel / 255) * (255 / f)))

            I[x, y] = new_pixel

            error_pixel = pixel - new_pixel

            fs_pixel = I[x + 1, y] + round(error_pixel * (7 / 16))

            I[x + 1, y] = fs_pixel

            fs_pixel = I[x - 1, y + 1] + round(error_pixel * 3 / 16)

            I[x - 1, y + 1] = fs_pixel

            fs_pixel = I[x, y + 1] + round(error_pixel * 5 / 16)

            I[x, y + 1] = fs_pixel

            fs_pixel = I[x + 1, y + 1] + round(error_pixel * 1 / 16)

            I[x + 1, y + 1] = fs_pixel
    return I


original_I = imageio.imread('vickson-santos-282.jpeg')

plt.subplot(2,2,1)
plt.title("Orginal")
plt.imshow(original_I, cmap="gray")
print("Inläsning klar")


halftone_I = halftone_function(original_I, )
plt.subplot(2,2,2)
plt.title("Halftone")
halftone_I = np.clip(halftone_I, 0, 255).astype('uint8')
plt.imshow(halftone_I, cmap="gray")
print("Halftone klar")


bw_fs_I = FS_bw(original_I, 1)
plt.subplot(2,2,3)
plt.title("FS Svartvit")
bw_fs_I = np.clip(bw_fs_I, 0, 255).astype('uint8')
plt.imshow(bw_fs_I, cmap="gray")
print("BW klar")



color_fs_I = floyd_steinberg_color(original_I, 1)
plt.subplot(2,2,4)
plt.title("FS Färgbild")
color_fs_I = np.clip(color_fs_I, 0, 255).astype('uint8')
plt.imshow(color_fs_I)
print("Color klar")


imageio.imwrite('picture_out_halftone.jpg', halftone_I)
imageio.imwrite('picture_out_fsbw.jpg', bw_fs_I)
imageio.imwrite('picture_out_fsrgb.jpg', color_fs_I)



plt.show()
