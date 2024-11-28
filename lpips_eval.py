import lpips
import torch
from PIL import Image
from torchvision import transforms
import os

# Load LPIPS model
loss_fn = lpips.LPIPS(net='alex')  # Options: 'alex', 'vgg'

# Load and preprocess images
def load_image(path):
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    return transform(Image.open(path)).unsqueeze(0)

# Get list of prompts from real_images folder
prompts = [f[:-9] for f in os.listdir("real_images") if (os.path.isfile(os.path.join("real_images", f)) and f != ".DS_Store")]
num_prompts = len(prompts)
total_scores = 0

# Run the model against each prompt
for image in prompts:
    real_image = load_image("real_images/" + image + "_real.png")
    generated_image = load_image("generated_images/" + image + "_generated.png")
    distance = loss_fn(real_image, generated_image)
    total_scores += distance.item()

# Compute Average LPIPS
average = total_scores / num_prompts
print(f"Average LPIPS Distance: {average}") # returns 0.495188112060229




