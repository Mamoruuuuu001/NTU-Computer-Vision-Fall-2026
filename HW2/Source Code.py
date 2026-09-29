import cv2
import numpy as np
from openpyxl import Workbook
import matplotlib.pyplot as plt

# (a) a binary image (threshold at 128)
def binarize(img):
    h, w = img.shape
    binary = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            binary[i][j] = 255 if img[i][j] >= 128 else 0

    return binary

 
# (b) a histogram
def compute_histogram(img):
    hist = [0] * 256

    for row in img:
        for pixel in row:
            hist[pixel] += 1

    return hist


# histogram
def save_histogram(hist):
    # excel spreadsheet
    wb = Workbook()
    ws = wb.active
    ws.title = "Histogram"

    ws.append(["Intensity", "Count"])
    for i, val in enumerate(hist):
        ws.append([i, val])

    wb.save("histogram.xlsx")

    # Save histogram image
    plt.bar(range(256), hist)
    plt.title("lena.bmp histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.savefig("histogram.png")
    plt.clf()

 
# (c) connected components (regions with + at centroid, bounding box)
def connected_components(binary):
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for i in range(h):
        for j in range(w):
            if binary[i][j] == 255 and not visited[i][j]:
                stack = [(i, j)]
                visited[i][j] = True
                pixels = []

                while stack:
                    x, y = stack.pop()
                    pixels.append((x, y))

                    for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                        if 0 <= nx < h and 0 <= ny < w:
                            if binary[nx][ny] == 255 and not visited[nx][ny]:
                                visited[nx][ny] = True
                                stack.append((nx, ny))

                if len(pixels) >= 500:
                    components.append(pixels)

    return components

def get_color(idx, total):
    h = 6.0 * idx / max(total, 1)
    s, v = 0.78, 0.86
    i = int(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t),
               (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return (int(b * 255), int(g * 255), int(r * 255))  


def draw_components(img, components):
    color_img = np.stack([img, img, img], axis=2) 

    for idx, comp in enumerate(components):
        color = get_color(idx, len(components))

        xs, ys = zip(*comp)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        for x, y in comp:
            color_img[x, y] = color

        cv2.rectangle(color_img, (min_y, min_x), (max_y, max_x), color, 2)

        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))

        cv2.line(color_img, (cy-10, cx), (cy+10, cx), (255,255,255), 2)
        cv2.line(color_img, (cy, cx-10), (cy, cx+10), (255,255,255), 2)

    return color_img



img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

binary = binarize(img)
cv2.imwrite("binary.bmp", binary)

hist = compute_histogram(img)
save_histogram(hist)

components = connected_components(binary)
result = draw_components(binary, components)
cv2.imwrite("connected_components.bmp", result)

print("Connected components (>=500 pixels):", len(components))