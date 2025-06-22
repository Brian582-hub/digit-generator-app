
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Model definition (must match the architecture used during training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# Load model
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("models/cgan_generator.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

generator = load_model()

# UI
st.title("Handwritten Digit Generator (cGAN)")
st.markdown("Select a digit (0–9) and generate 5 handwritten digit images.")

digit = st.selectbox("Select a digit", list(range(10)))
generate_button = st.button("Generate Images")

if generate_button:
    noise = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)

    with torch.no_grad():
        gen_imgs = generator(noise, labels)

    # Plot and display images
    st.write(f"Generated Images for Digit: {digit}")
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        img = gen_imgs[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)
