import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema

# def debug_image_processing(image_path, threshold=70):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError("Image not found")

#     # Threshold (try both!)
#     _, bw_inv = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
#     _, bw = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

#     # Find contours
#     contours_inv, _ = cv2.findContours(
#         bw_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
#     )
#     contours, _ = cv2.findContours(
#         bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
#     )

#     # Draw contours
#     img_cont_inv = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.drawContours(img_cont_inv, contours_inv, -1, (0, 0, 255), 1)

#     img_cont = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.drawContours(img_cont, contours, -1, (0, 0, 255), 1)

#     # --- Plot everything ---
#     fig, axs = plt.subplots(2, 3, figsize=(14, 8))

#     axs[0, 0].imshow(img, cmap="gray")
#     axs[0, 0].set_title("Original grayscale")

#     axs[0, 1].imshow(bw, cmap="gray")
#     axs[0, 1].set_title("Binary")

#     axs[0, 2].imshow(bw_inv, cmap="gray")
#     axs[0, 2].set_title("Binary inverted")

#     axs[1, 0].imshow(img_cont)
#     axs[1, 0].set_title(f"Contours (binary) | N={len(contours)}")

#     axs[1, 1].imshow(img_cont_inv)
#     axs[1, 1].set_title(f"Contours (inverted) | N={len(contours_inv)}")

#     axs[1, 2].axis("off")

#     for ax in axs.flat:
#         ax.axis("off")

#     plt.tight_layout()
#     plt.show()

# debug_image_processing("fraun.png")

import cv2
import numpy as np

def intensity_band_mask(image_path, dI=0.01):
    # Read grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found")

    # Normalize to [0, 1]
    img_f = img.astype(np.float32) / 255.0
    I_max = img_f.max()


    # Band-pass mask
    mask = np.logical_and(
        img_f >= (I_max - dI),
        img_f <= (I_max + dI)
    )

    # Convert to uint8 image (0 or 255)
    mask = (mask.astype(np.uint8)) * 255

    return img_f, mask

img_f, mask = intensity_band_mask("fraun.png", dI=0.04)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img_f, cmap="gray")
plt.title("Normalized grayscale")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mask, cmap="gray")
plt.title("Intensity band mask")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img_f, cmap="gray")
plt.imshow(mask, cmap="jet", alpha=0.6)
plt.title("Mask overlay")
plt.axis("off")

plt.tight_layout()
plt.show()


def mask_to_function(mask):
    """
    mask: binary image (0 or 255), shape (H, W)
    returns: x array, y(x) array
    """
    H, W = mask.shape

    xs = []
    ys = []

    for x in range(W):
        # Find y-indices where mask is white
        y_vals = np.where(mask[:, x] > 0)[0]

        if len(y_vals) == 0:
            continue  # no data in this column

        y_mean = np.mean(y_vals)

        xs.append(x)
        ys.append(y_mean)

    xs = np.array(xs)
    ys = np.array(ys)

    # Convert image coords → Cartesian coords
    ys = H - ys

    return xs, ys

x, y = mask_to_function(mask)
y = savgol_filter(y, 3, 2)

# def find_local_minima(y):
#     """
#     Returns indices of strict local minima
#     """
#     minima = []
#     for i in range(1, len(y) - 1):
#         if y[i] < y[i - 1] and y[i] < y[i + 1]:
#             minima.append(i)
#     return minima

def find_local_minima(y,gap,bar_height):
    minima=[]
    for i in range(1,len(y)-1):
        if y[i] <= y[i-1] and y[i] <= y[i+1]:
            print(i)
            for j in range(i+1,min(len(y)-1,i+gap)):
                if y[j] >= y[j-1] and y[j] >= y[j+1] and (y[j]-y[i]) > bar_height:
                    print(i,j)
                    minima.append(i)
                    break
    return minima

def rescale_around_minima(y, window=2):
    """
    y: 1D numpy array
    window: half-width of rescaling window
    """
    y2 = y.copy()
    N = len(y)

    minima = find_local_minima(y,20,16)

    for i in minima:
        L = max(0, i - window)
        R = min(N - 1, i + window)

        yL = y[L]
        yR = y[R]
        y0 = y[i]

        # Skip degenerate cases
        if yL == y0 or yR == y0:
            continue

        # Left side: map y0 → 0, yL → yL
        for j in range(L, i + 1):
            # y2[j] = (y[j] - y0) * (yL / (yL - y0))
            y2[j] = y[j] - y0*(j-L)/(i-L)

        # Right side: map y0 → 0, yR → yR
        for j in range(i, R + 1):
            # y2[j] = (y[j] - y0) * (yR / (yR - y0))
            y2[j] = y[j] - y0*(j-R)/(i-R)

    return y2

# minima = find_local_minima(y,30,20)
y_new = rescale_around_minima(y, window=3)

def sinc_to_sin(y):
    y2=y.copy()
    fact=-1
    for i in range(len(y2)):
        y2[i]=fact*(y[i]**0.5)
        if y2[i]==0:
            fact*=-1
    return y2

y_final=sinc_to_sin(y_new)

# plt.plot(x, y, color='black')
plt.plot(x,y_final, color='red')
# plt.scatter(minima, y_new[minima], color='blue')
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.xticks(np.arange(0, np.max(x)+1, 50))
plt.title("Extracted boundary: y = f(x)")
plt.grid(True, alpha=0.3)
plt.show()


# plt.figure(figsize=(6,4))
# plt.imshow(vis)
# plt.axis("off")
# plt.show()


# def image_to_function(image_path, smooth=True):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError("Image not found")

#     h, w = img.shape

#     # Optional smoothing to suppress vertical streaks
#     if smooth:
#         img = cv2.GaussianBlur(img, (5, 5), 0)

#     # Vertical gradient (detect horizontal edges)
#     grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
#     grad_y = np.abs(grad_y)

#     xs = np.arange(w)
#     ys = np.zeros(w)

#     for x in range(160,w-155):
#         column = grad_y[:, x]

#         # Ignore borders
#         column[:10] = 0
#         column[-10:] = 0

#         # Strongest vertical edge
#         ys[x] = np.argmax(column)

#     # Convert image coords → Cartesian
#     c=0
#     for i in ys:
#         if i!=0:
#             ys[c] = h - i
#         else:
#             ys[c] = h/100
#         c+=1

#     # if len(ys) > 51:
#     #     ys = savgol_filter(ys, 10, 3)

#     return xs, ys


# x, y = image_to_function("fraun.png")

# def plot_extracted_function(x, y):
#     plt.figure(figsize=(8, 4))
#     plt.plot(x, y, color='black', linewidth=2)
#     plt.xlabel("x (pixels)")
#     plt.ylabel("y (pixels)")
#     plt.title("Extracted boundary: y = f(x)")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()

# # plot_extracted_function(x, y)


# ----- define domain -----
print(np.size(x))
N = 685          # number of points
# L = 50          # domain size
# dx = L / N
# x = np.linspace(-L/2, L/2, N)

# def ddf0(x,sig): 
#     val = np.zeros_like(x)
#     val[(-(1/(2*sig))<=x) & (x<=(1/(2*sig)))] = 1
#     return val

# def ddf(x, sigma):
#     return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-x**2 / (2*sigma**2))

# a=0.333
# sigma=0.5
# dist=20
# # ----- DEFINE YOUR FUNCTION HERE -----
# def f(x):
#     return a*ddf(x-dist,sigma) + ddf(x+dist,sigma) + 0.1*ddf0(x+dist,0.2) + a*0.1*ddf0(x-dist,0.2) # example: Gaussian

fx = y_final
L = np.size(y_final)
x = np.linspace(-L/2, L/2, N)
dx=L/N


# ----- Fourier transform -----
F = np.fft.fft(fx)
F = np.fft.fftshift(F)

# corresponding k values
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
k = np.fft.fftshift(k)

# ----- plot -----
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(x, fx)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Function")

ax2 = plt.subplot(1,2,2)
ax2.set_facecolor('red') 
plt.plot(k, np.abs(F),color='white', linewidth=2)
plt.fill_between(k, np.abs(F), 0, color='blue')
# solid white border
# rect = Rectangle(
#     (0, 0), 1, 1,
#     transform=ax2.transAxes,
#     fill=False,
#     edgecolor='white',
#     linewidth=1.5,
#     zorder=10
# )
# ax2.add_patch(rect)

# # fake smudge / glow (second, thicker, transparent rectangle)
# glow = Rectangle(
#     (0, 0), 1, 1,
#     transform=ax2.transAxes,
#     fill=False,
#     edgecolor='white',
#     linewidth=6,
#     alpha=0.15,
#     zorder=9
# )
# ax2.add_patch(glow)

# plt.fill_between(k, F, alpha=1)
plt.xlabel("k")
plt.ylabel(r"$|\tilde f(k)|$")
plt.xlim(-1,1)
plt.title("Fourier Transform")

plt.tight_layout()
plt.show()
