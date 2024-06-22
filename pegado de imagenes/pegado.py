import os
from PIL import Image

# Define folder paths
base_dir = os.path.dirname(os.path.abspath(__file__))
carpeta1 = os.path.join(base_dir, 'carpeta1')
carpeta2 = os.path.join(base_dir, 'carpeta2')
carpeta3 = os.path.join(base_dir, 'carpeta3')

# Create carpeta3 if it doesn't exist
if not os.path.exists(carpeta3):
    os.makedirs(carpeta3)

# Get list of image files in carpeta1 and carpeta2
carpeta1_images = [f for f in os.listdir(carpeta1) if os.path.isfile(os.path.join(carpeta1, f))]
carpeta2_images = [f for f in os.listdir(carpeta2) if os.path.isfile(os.path.join(carpeta2, f))]

# Iterate through each image in carpeta1 and carpeta2
for img1_name in carpeta1_images:
    img1_path = os.path.join(carpeta1, img1_name)
    img1 = Image.open(img1_path).convert("RGBA")

    for img2_name in carpeta2_images:
        img2_path = os.path.join(carpeta2, img2_name)
        img2 = Image.open(img2_path).convert("RGBA")

        # Resize img1 to match img2 size
        img1_resized = img1.resize(img2.size)

        # Composite images
        combined = Image.alpha_composite(img2, img1_resized)

        # Save the result in carpeta3
        combined_name = f"{os.path.splitext(img1_name)[0]}_on_{os.path.splitext(img2_name)[0]}.png"
        combined_path = os.path.join(carpeta3, combined_name)
        combined.save(combined_path, "PNG")

print("finalizado")
