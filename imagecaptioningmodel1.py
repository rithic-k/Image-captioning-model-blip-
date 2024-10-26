from google.colab import drive
drive.mount('/content/drive')

import kagglehub

# Download latest version
path = kagglehub.dataset_download("dataclusterlabs/vehicle-image-captioning-dataset")

print("Path to dataset files:", path)



import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Set the path to the dataset in Google Drive
image_dir = '/content/drive/MyDrive/vehicle-image-captioning-dataset/indian_vehicle_images/indian_vehicle_images'  # Adjust as necessary
captions = {}

# Load model and processor
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

# Loop through images and generate captions
for image_name in os.listdir(image_dir):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)

        # Prepare the image
        inputs = processor(images=image, return_tensors='pt')

        # Generate caption
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        captions[image_name] = caption

# Display captions
for img, cap in captions.items():
    print(f'{img}: {cap}')

# Save captions to a text file in Google Drive
with open('/content/drive/MyDrive/captions.txt', 'w') as f:
    for img, cap in captions.items():
        f.write(f'{img}: {cap}\n')

print('Captions saved to /content/drive/MyDrive/captions.txt')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Load the captions CSV file
#captions_df = pd.read_csv('/content/drive/My Drive/vehicle-image-captioning-dataset/captions.csv')
#************************above line is commented because its not necessary but if the lower portion doesnt work, uncomment it ***********
# Function to display image and caption
def display_image_caption():
  """Displays an image and its corresponding caption.

  Args:
    image_id: The ID of the image to display.
  """
  image_path = '/content/drive/MyDrive/vehicle-image-captioning-dataset/indian_vehicle_images/indian_vehicle_images'
  images = os.listdir(image_path)
  image_id = images[0]
  print(image_id)
  caption = captions[image_id]

  img_path=os.path.join(image_path, image_id)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('off')  # Turn off axis numbers and ticks
  plt.title(caption)
  plt.show()

# Example usage: Display image with ID '1'
display_image_caption()

caption = captions[['image_id'] == image_id]['caption'].values[0]