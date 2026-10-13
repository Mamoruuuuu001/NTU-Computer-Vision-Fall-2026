import sys
import numpy as np
import cv2

# Octagonal 3-5-5-5-3 kernel, origin at the center
OCTAGON = [(dy, dx)
           for dy, half in [(-2, 1), (-1, 2), (0, 2), (1, 2), (2, 1)]
           for dx in range(-half, half + 1)]

# "L" shaped kernels for the upper-right corner
J_KERNEL = [(0, 0), (0, -1), (1, 0)]
K_KERNEL = [(-1, 0), (-1, 1), (0, 1)]

def shift(img, dy, dx):
    h, w = img.shape
    out = np.zeros_like(img)
    ys, ye = max(0, -dy), min(h, h - dy)
    xs, xe = max(0, -dx), min(w, w - dx)
    out[ys:ye, xs:xe] = img[ys + dy:ye + dy, xs + dx:xe + dx]
    return out

# (a) Dilation
def dilation(img, kernel):
    out = np.zeros_like(img)
    for dy, dx in kernel:
        out |= shift(img, -dy, -dx)
    return out

# (b) Erosion
def erosion(img, kernel):
    out = np.ones_like(img)
    for dy, dx in kernel:
        out &= shift(img, dy, dx)
    return out

# (c) Opening
def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)


# (d) Closing
def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

# (e) Hit-and-miss transform
def hit_and_miss(img, j_kernel, k_kernel):
    return erosion(img, j_kernel) & erosion(1 - img, k_kernel)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "lena.bmp"
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"Cannot read {path}")

    # binarize image
    binary = (gray >= 128).astype(np.uint8)

    results = {
        "binary":   binary,
        "dilation": dilation(binary, OCTAGON),
        "erosion":  erosion(binary, OCTAGON),
        "opening":  opening(binary, OCTAGON),
        "closing":  closing(binary, OCTAGON),
        "hit_and_miss": hit_and_miss(binary, J_KERNEL, K_KERNEL),
    }
    for name, res in results.items():
        cv2.imwrite(f"{name}.bmp", res * 255)
        print(f"saved {name}.bmp  (white pixels: {int(res.sum())})")


if __name__ == "__main__":
    main()