import cv2
import numpy as np
from openpyxl import Workbook
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# histogram (no np.histogram)
def compute_histogram(img):
    hist = [0] * 256

    for row in img:
        for pixel in row:
            hist[pixel] += 1

    return hist


# save histogram as excel
def save_histogram(hist, name, title):
    # excel spreadsheet
    wb = Workbook()
    ws = wb.active
    ws.title = "Histogram"

    ws.append(["Intensity", "Count"])
    for i, val in enumerate(hist):
        ws.append([i, val])

    wb.save(name + ".xlsx")

    # histogram image
    fig = plt.figure(figsize=(6, 4), dpi=100)
    plt.bar(range(256), hist, width=1.0)
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.xlim(0, 255)
    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    cv2.imwrite(name + ".bmp", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))


# (b) image with intensity divided by 3 and its histogram
def divide_by_3(img):
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            out[i][j] = int(img[i][j]) // 3

    return out


# (c) image after applying histogram equalization to (b) and its histogram
def equalize(img, hist):
    h, w = img.shape
    total = h * w

    cdf = np.cumsum(hist)
    cdf_min = 0
    for v in range(256):
        if hist[v] != 0:
            cdf_min = cdf[v]
            break

    lut = [0] * 256
    for v in range(256):
        if total == cdf_min:
            lut[v] = v
        else:
            lut[v] = int((cdf[v] - cdf_min) / (total - cdf_min) * 255 + 0.5)
            lut[v] = max(0, min(255, lut[v]))

    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i][j] = lut[img[i][j]]

    return out


img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

# (a) original image and its histogram
hist_a = compute_histogram(img)
cv2.imwrite("a_original.bmp", img)
save_histogram(hist_a, "a_histogram", "(a) original histogram")

# (b) (b) image with intensity divided by 3 and its histogram
img_b = divide_by_3(img)
hist_b = compute_histogram(img_b)
cv2.imwrite("b_div3.bmp", img_b)
save_histogram(hist_b, "b_histogram", "(b) intensity / 3 histogram")

# (c) image after applying histogram equalization to (b) and its histogram
img_c = equalize(img_b, hist_b)
hist_c = compute_histogram(img_c)
cv2.imwrite("c_equalized.bmp", img_c)
save_histogram(hist_c, "c_histogram", "(c) equalized histogram")

print("Done: a_original / b_div3 / c_equalized + their histograms (.bmp, .xlsx)")