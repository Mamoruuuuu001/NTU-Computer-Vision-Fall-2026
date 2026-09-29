import cv2

img = cv2.imread('lena.bmp')
height, width, channels = img.shape

# (a) upside-down lena.bmp
upside_down = img.copy()
for i in range(height):
    for j in range(width):
        upside_down[i][j] = img[height - 1 - i][j]

cv2.imwrite('upside-down lena.bmp', upside_down)


# (b) right-side-left lena.bmp
right_left = img.copy()
for i in range(height):
    for j in range(width):
        right_left[i][j] = img[i][width - 1 - j]

cv2.imwrite('right-side-left lena.bmp', right_left)


# (c) diagonally flip lena.bmp
diagonal = img.copy()
for i in range(height):
    for j in range(width):
        diagonal[i][j] = img[j][i]

cv2.imwrite('diagonally flip lena.bmp', diagonal)