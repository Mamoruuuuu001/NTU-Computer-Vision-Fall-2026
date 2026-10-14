import numpy as np
import cv2

# Octagonal 3-5-5-5-3 kernel, origin at the center
OCTAGON = [(dy, dx)
           for dy, half in [(-2, 1), (-1, 2), (0, 2), (1, 2), (2, 1)]
           for dx in range(-half, half + 1)]

def shift(img, dy, dx, fill):
    h, w = img.shape
    out = np.full_like(img, fill)
    ys, ye = max(0, -dy), min(h, h - dy)
    xs, xe = max(0, -dx), min(w, w - dx)
    out[ys:ye, xs:xe] = img[ys + dy:ye + dy, xs + dx:xe + dx]
    return out

# (a) Dilation
def dilation(img, kernel):
    out = np.zeros_like(img)
    for dy, dx in kernel:
        out = np.maximum(out, shift(img, -dy, -dx, 0))
    return out

# (b) Erosion
def erosion(img, kernel):
    out = np.full_like(img, 255)
    for dy, dx in kernel:
        out = np.minimum(out, shift(img, dy, dx, 255))
    return out

# (c) Opening
def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)

# (d) Closing
def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

def main():
    gray = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("Cannot read lena.bmp")
        exit()

    results = {
        "gray":   gray,
        "dilation": dilation(gray, OCTAGON),
        "erosion":  erosion(gray, OCTAGON),
        "opening":  opening(gray, OCTAGON),
        "closing":  closing(gray, OCTAGON),
    }
    for name, res in results.items():
        cv2.imwrite(f"{name}.bmp", res)
        print(f"saved {name}.bmp")

if __name__ == "__main__":
    main()