from llava.model.builder import load_pretrained_model
import os
import torch
from urllib.request import urlopen
from PIL import Image

# Up to 10 minute load time
cwd = os.path.dirname(os.getcwd())
print(cwd)
model_path = os.path.join(cwd, "llava-med-v1.5-mistral-7b")
data_path = os.path.join(cwd, "llava_med_eval_qa50_fig_captions.json")

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name='llava-med-v1.5-mistral-7b'
)

template = 'this is a photo of '
labels = [
    'adenocarcinoma histopathology',
    'brain MRI',
    'covid line chart',
    'squamous cell carcinoma histopathology',
    'immunohistochemistry histopathology',
    'bone X-ray',
    'chest X-ray',
    'pie chart',
    'hematoxylin and eosin histopathology'
]
dataset_url = 'https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/'
test_imgs = [
    'squamous_cell_carcinoma_histopathology.jpeg',
    'H_and_E_histopathology.jpg',
    'bone_X-ray.jpg',
    'adenocarcinoma_histopathology.jpg',
    'covid_line_chart.png',
    'IHC_histopathology.jpg',
    'chest_X-ray.jpg',
    'brain_MRI.jpg',
    'pie_chart.png'
]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ]
    }
]

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Running on CUDA")
else:
    print("Running on CPU")

model.to(device)
model.eval()

images = [image_processor(Image.open(urlopen(dataset_url + img)), do_normalize=False)["pixel_values"] for img in test_imgs]
images = (255 * torch.tensor(images).flatten(0, 1)).to(dtype=torch.long).to(device)
print(images.shape)
print(images)

texts = tokenizer([template + file for file in labels], padding=True, truncation=True)
texts = torch.tensor(texts["input_ids"]).to(dtype=torch.long).to(device)
print(texts)

# Debug
print(f"texts dtype: {texts.dtype}")
print(f"images dtype: {images.dtype}")
print(f"Images batch size: {images.shape[0]}")
print(f"Texts batch size: {texts.shape[0]}")  # Should match images batch size

print(f"Model input types: images={images.dtype}, texts={texts.dtype}")
print(f"Images shape: {images.shape}, Texts shape: {texts.shape}")
print(f"Images max value: {images.max().item()}")
print(f"Images min value: {images.min().item()}")

print(texts.shape, images.shape)

"""
# For doing one entry at a time (reindent the below section)

for i in range(texts.shape[0]):
    text = texts[i]
    image = images[i]
    print(text.shape, image.shape)
"""

with torch.no_grad():
    image_features, text_features, logit_scale = model(images, texts)

    logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()

top_k = -1

for i, img in enumerate(test_imgs):
    pred = labels[sorted_indices[i][0]]

    top_k = len(labels) if top_k == -1 else top_k
    print(img.split('/')[-1] + ':')
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        print(f'{labels[jth_index]}: {logits[i][jth_index]}')
    print('\n')
