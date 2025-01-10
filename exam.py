import numpy as np
import matplotlib.pyplot as plt

def F(x):
    return 255 * np.exp(-x**2)

def G(x, y):
    return np.cos(20 * np.arccos(np.sqrt(x**2 + y**2) / 255))

def B(x, y):
    return np.sin((x**2 + y**2) / 200)

def generate_image(width, height):
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Normalize coordinates
            nx = (x - width / 2) / (width / 2)
            ny = (y - height / 2) / (height / 2)

            # Generate H values for RGB channels
            H_r = nx * G(nx, ny) + B(nx, ny)
            H_g = ny * G(nx, ny) + B(nx, ny)
            H_b = nx * G(nx, ny) - B(nx, ny)

            # Map H values to RGB using F
            image[y, x, 0] = np.clip(F(H_r), 0, 255)  # Red
            image[y, x, 1] = np.clip(F(H_g), 0, 255)  # Green
            image[y, x, 2] = np.clip(F(H_b), 0, 255)  # Blue

    return image

# Set image dimensions
width = 800
height = 600

# Generate and display the image
image = generate_image(width, height)

plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.axis('off')
plt.show()
