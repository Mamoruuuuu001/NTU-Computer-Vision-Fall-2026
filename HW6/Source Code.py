import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

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

# Yokoi connectivity number helper function h
# h(b, c, d, e):
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

# Yokoi connectivity number f
# f(a1, a2, a3, a4):
#   5 if all are 'r'
#   count of 'q' otherwise
def f_func(a1, a2, a3, a4):
    if a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r':
        return 5
    else:
        return [a1, a2, a3, a4].count('q')

# Get pixel value safely (0 for out of bounds)
def get_pixel(img, i, j):
    h, w = img.shape
    if i < 0 or i >= h or j < 0 or j >= w:
        return 0
    return img[i][j]

# Compute Yokoi connectivity number for the entire image
def yokoi_connectivity(img):
    h, w = img.shape
    result = np.full((h, w), -1, dtype=int)  # -1 means background

    for i in range(h):
        for j in range(w):
            # Only compute for foreground pixels (value 255)
            if img[i][j] == 0:
                continue

            # x0 = center (i,j), neighbors in 4-connected + diagonals
            # x0 = (i,j), x1 = (i,j+1), x2 = (i-1,j), x3 = (i,j-1), x4 = (i+1,j)
            # x5 = (i-1,j+1), x6 = (i-1,j-1), x7 = (i+1,j-1), x8 = (i+1,j+1)
            x0 = img[i][j]
            x1 = get_pixel(img, i, j + 1)      # right
            x2 = get_pixel(img, i - 1, j)      # top
            x3 = get_pixel(img, i, j - 1)      # left
            x4 = get_pixel(img, i + 1, j)      # bottom
            x5 = get_pixel(img, i - 1, j + 1)  # top-right
            x6 = get_pixel(img, i - 1, j - 1)  # top-left
            x7 = get_pixel(img, i + 1, j - 1)  # bottom-left
            x8 = get_pixel(img, i + 1, j + 1)  # bottom-right

            # Compute h for four quadrants
            a1 = h_func(x0, x1, x5, x2)  # quadrant 1: right, top-right, top
            a2 = h_func(x0, x2, x6, x3)  # quadrant 2: top, top-left, left
            a3 = h_func(x0, x3, x7, x4)  # quadrant 3: left, bottom-left, bottom
            a4 = h_func(x0, x4, x8, x1)  # quadrant 4: bottom, bottom-right, right

            result[i][j] = f_func(a1, a2, a3, a4)

    return result

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

    # Step 3: Compute Yokoi connectivity number
    yokoi = yokoi_connectivity(downsampled)

    # Step 4: Build the 64x64 matrix text
    h, w = yokoi.shape
    lines = []
    for i in range(h):
        row = ""
        for j in range(w):
            if yokoi[i][j] == -1:
                row += " "  # background: leave blank
            else:
                row += str(yokoi[i][j])
        lines.append(row)
        print(row)

    # Save as text file
    with open("yokoi_matrix.txt", "w") as f:
        for line in lines:
            f.write(line + "\n")

    # Step 5: Generate A4-sized image of the matrix
    # Render each cell individually to ensure perfect monospace alignment
    rows = len(lines)
    cols = 64
    cell_w = 9   # pixels per cell width
    cell_h = 11  # pixels per cell height
    margin = 20  # margin around the matrix

    img_w = cols * cell_w + 2 * margin
    img_h = rows * cell_h + 2 * margin

    # Create white image
    matrix_img = np.ones((img_h, img_w), dtype=np.uint8) * 255

    # Draw each character using cv2.putText
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1

    for i in range(rows):
        for j in range(cols):
            ch = lines[i][j] if j < len(lines[i]) else ' '
            if ch != ' ':
                x = margin + j * cell_w + 1
                y = margin + i * cell_h + cell_h - 2
                cv2.putText(matrix_img, ch, (x, y), font, font_scale, 0, thickness, cv2.LINE_AA)

    cv2.imwrite("yokoi_matrix.png", matrix_img)

    print("\nYokoi connectivity number matrix saved to yokoi_matrix.txt and yokoi_matrix.png")

if __name__ == "__main__":
    main()