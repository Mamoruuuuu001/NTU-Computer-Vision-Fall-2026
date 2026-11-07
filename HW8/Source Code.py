import os
import math
import numpy as np
import cv2

SEED = 0
OUT_DIR = "result"

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


def dilation(img, kernel):
    out = np.zeros_like(img)
    for dy, dx in kernel:
        out = np.maximum(out, shift(img, -dy, -dx, 0))
    return out


def erosion(img, kernel):
    out = np.full_like(img, 255)
    for dy, dx in kernel:
        out = np.minimum(out, shift(img, dy, dx, 255))
    return out


def opening(img, kernel):
    return dilation(erosion(img, kernel), kernel)


def closing(img, kernel):
    return erosion(dilation(img, kernel), kernel)

def gaussian_noise(img, amplitude, rng):
    noisy = img.astype(np.float64) + amplitude * rng.normal(0.0, 1.0, img.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def salt_and_pepper_noise(img, prob, rng):
    u = rng.uniform(0.0, 1.0, img.shape)
    noisy = img.copy()
    noisy[u <= prob] = 0
    noisy[u >= 1.0 - prob] = 255
    return noisy


def neighborhood(img, k):
    # k*k shifted copies of the reflect-padded image (stacked on axis 0)
    h, w = img.shape
    r = k // 2
    padded = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT)
    return np.stack([padded[r + di:r + di + h, r + dj:r + dj + w]
                     for di in range(-r, r + 1) for dj in range(-r, r + 1)], axis=0)


def box_filter(img, k):
    mean = np.sum(neighborhood(img, k), axis=0, dtype=np.float64) / (k * k)
    return np.clip(np.rint(mean), 0, 255).astype(np.uint8)


def median_filter(img, k):
    return np.sort(neighborhood(img, k), axis=0)[(k * k) // 2]


def opening_then_closing(img):
    return closing(opening(img, OCTAGON), OCTAGON)


def closing_then_opening(img):
    return opening(closing(img, OCTAGON), OCTAGON)


def snr(original, processed):
    a = original.astype(np.float64) / 255.0
    b = processed.astype(np.float64) / 255.0
    n = a.size
    mu = np.sum(a) / n
    vs = math.sqrt(np.sum((a - mu) ** 2) / n)
    mu_n = np.sum(b - a) / n
    vn = math.sqrt(np.sum((b - mu_n - a) ** 2) / n)
    return 20.0 * math.log10(vs / vn)


def main():
    img = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Cannot read lena.bmp")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    noisy_images = [
        ("gaussian_10", gaussian_noise(img, 10, rng)),
        ("gaussian_30", gaussian_noise(img, 30, rng)),
        ("saltpepper_0.1", salt_and_pepper_noise(img, 0.1, rng)),
        ("saltpepper_0.05", salt_and_pepper_noise(img, 0.05, rng)),
    ]
    filters = [
        ("box_3x3", lambda x: box_filter(x, 3)),
        ("box_5x5", lambda x: box_filter(x, 5)),
        ("median_3x3", lambda x: median_filter(x, 3)),
        ("median_5x5", lambda x: median_filter(x, 5)),
        ("open_close", opening_then_closing),
        ("close_open", closing_then_opening),
    ]

    lines = []
    for noise_name, noisy in noisy_images:
        cv2.imwrite(os.path.join(OUT_DIR, noise_name + ".bmp"), noisy)
        lines.append("%-32s SNR = %.5f" % (noise_name, snr(img, noisy)))
        for filter_name, filter_func in filters:
            result = filter_func(noisy)
            name = noise_name + "_" + filter_name
            cv2.imwrite(os.path.join(OUT_DIR, name + ".bmp"), result)
            lines.append("%-32s SNR = %.5f" % (name, snr(img, result)))

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "snr.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()