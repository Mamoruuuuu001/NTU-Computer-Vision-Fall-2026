import numpy as np
import cv2
BACKGROUND = 0
Q = 1
P = 2

# Downsampling Lena from 512x512 to 64x64
def downsample(img, block_size=8):
    h, w = img.shape
    new_h = h // block_size
    new_w = w // block_size
    down = np.zeros((new_h, new_w), dtype=np.uint8)
    for i in range(new_h):
        for j in range(new_w):
            down[i][j] = img[i * block_size][j * block_size]
    return down


# Binarize lena
def binarize(img):
    h, w = img.shape
    binary = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            binary[i][j] = 255 if img[i][j] >= 128 else 0
    return binary


# Yokoi helper h(b, c, d, e) (4-connectivity)
#   q if b == c and (d != b or e != b)
#   r if b == c and d == b and e == b
#   s if b != c
def h_func(b, c, d, e):
    if b == c:
        if d != b or e != b:
            return 'q'
        else:
            return 'r'
    else:
        return 's'


# Yokoi connectivity number f(a1, a2, a3, a4)
#   5 if all are 'r', otherwise the number of 'q'
def f_func(a1, a2, a3, a4):
    if a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r':
        return 5
    else:
        return [a1, a2, a3, a4].count('q')


# Get pixel value safely (0 for out of bounds -> padding with background)
def get_pixel(img, i, j):
    h, w = img.shape
    if i < 0 or i >= h or j < 0 or j >= w:
        return 0
    return img[i][j]


# h results of the four quadrants around pixel (i, j)
def quadrant_labels(img, i, j):
    x0 = img[i][j]
    x1 = get_pixel(img, i, j + 1)      # right
    x2 = get_pixel(img, i - 1, j)      # top
    x3 = get_pixel(img, i, j - 1)      # left
    x4 = get_pixel(img, i + 1, j)      # bottom
    x5 = get_pixel(img, i + 1, j + 1)  # bottom-right
    x6 = get_pixel(img, i - 1, j + 1)  # top-right
    x7 = get_pixel(img, i - 1, j - 1)  # top-left
    x8 = get_pixel(img, i + 1, j - 1)  # bottom-left

    a1 = h_func(x0, x1, x6, x2)
    a2 = h_func(x0, x2, x7, x3)
    a3 = h_func(x0, x3, x8, x4)
    a4 = h_func(x0, x4, x5, x1)
    return a1, a2, a3, a4


# Compute Yokoi connectivity number (-1 for background)
def yokoi_connectivity(img):
    h, w = img.shape
    result = np.full((h, w), -1, dtype=int)
    for i in range(h):
        for j in range(w):
            if img[i][j] == 0:
                continue
            a1, a2, a3, a4 = quadrant_labels(img, i, j)
            result[i][j] = f_func(a1, a2, a3, a4)
    return result

# Pair relationship operator on the Yokoi result.
#   h(a, m) = 1 if a == m else 0
#   output p if  sum_{k=1..4} h(yokoi(x_k), 1) >= 1  and  yokoi(x0) == 1
#   output q otherwise (for foreground pixels)
def pair_relationship(yokoi):
    h, w = yokoi.shape
    marked = np.full((h, w), BACKGROUND, dtype=np.uint8)
    neighbors = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # 4-neighbors
    for i in range(h):
        for j in range(w):
            if yokoi[i][j] == -1:
                continue
            count = 0
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and yokoi[ni][nj] == 1:
                    count += 1
            if yokoi[i][j] == 1 and count >= 1:
                marked[i][j] = P
            else:
                marked[i][j] = Q
    return marked


# Connected shrink operator for a single pixel: it may be removed when its
# Yokoi connectivity number is exactly 1 (removal keeps the object connected).
def can_shrink(img, i, j):
    a1, a2, a3, a4 = quadrant_labels(img, i, j)
    return f_func(a1, a2, a3, a4) == 1


# One full pass: Yokoi -> pair relationship -> connected shrink (in place)
def thinning_pass(img):
    h, w = img.shape
    yokoi = yokoi_connectivity(img)
    marked = pair_relationship(yokoi)

    changed = False
    for i in range(h):
        for j in range(w):
            if marked[i][j] != P:
                continue
            # evaluated on the current (already updated) image
            if img[i][j] != 0 and can_shrink(img, i, j):
                img[i][j] = 0
                changed = True
    return changed


# Iterate until the image no longer changes
def thinning(img):
    thin = img.copy()
    iterations = 0
    while thinning_pass(thin):
        iterations += 1
    return thin, iterations


def main():
    # Read image
    gray = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("Cannot read lena.bmp")
        return

    # Step 1: Binarize (threshold at 128)
    binary = binarize(gray)

    # Step 2: Downsample from 512x512 to 64x64
    downsampled = downsample(binary)

    # Step 3: Thinning (4-connected)
    thinned, iterations = thinning(downsampled)
    print("Thinning finished after %d iterations that changed the image" % iterations)

    # Save the 64x64 thinned result
    cv2.imwrite("thinning.bmp", thinned)
    print("saved thinning.bmp")


if __name__ == "__main__":
    main()