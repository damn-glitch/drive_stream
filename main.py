import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------
# Dummy PyTorch Model
# --------------------------------------------------------------------
class DriveVisionModel(nn.Module):
    def __init__(self):
        super(DriveVisionModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc   = nn.Linear(16*64*64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --------------------------------------------------------------------
# Generate a placeholder image (draws text in the center)
# --------------------------------------------------------------------
def generate_placeholder_image(label_text, size=(600, 300), bg_color=(220, 220, 220)):
    """
    Creates a basic PIL image with a centered text label.
    """
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Use font.getsize() instead of draw.textsize()
    text_w, text_h = font.getsize(label_text)
    pos = ((size[0] - text_w) // 2, (size[1] - text_h) // 2)
    draw.text(pos, label_text, fill=(0, 0, 0), font=font)
    return img

# --------------------------------------------------------------------
# Dummy data for demonstration
# --------------------------------------------------------------------
dummy_data = pd.DataFrame({
    "Time": np.arange(10),
    "Sensor1": np.random.randn(10),
    "Sensor2": np.random.randn(10),
    "Sensor3": np.random.randn(10)
})

# Instantiate a dummy model
model = DriveVisionModel()
fake_input = torch.randn(1, 3, 64, 64)

# --------------------------------------------------------------------
# Main App
# --------------------------------------------------------------------
def main():
    st.set_page_config(page_title="DriveVision Technical App", layout="wide")

    menu = ["Project Architecture", "ML Pipeline", "Live Demo (Simulation)", "Data Visualization"]
    choice = st.sidebar.radio("Navigate", menu)

    if choice == "Project Architecture":
        arch_page()
    elif choice == "ML Pipeline":
        ml_pipeline_page()
    elif choice == "Live Demo (Simulation)":
        live_demo_page()
    elif choice == "Data Visualization":
        data_viz_page()

# --------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------
def arch_page():
    col1, col2 = st.columns(2)
    with col1:
        st.title("DriveVision Technical Architecture")
        st.write("Unified approach: Perception, Prediction, Planning in one pipeline.")
        st.write("Query-based design linking modules to a single planning-oriented model.")
    with col2:
        img = generate_placeholder_image("High-Level System Architecture")
        st.image(img)

    st.write("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Perception")
        st.write("• Object Detection")
        st.write("• Environment Mapping")
        st.write("• Occupancy Grid Updates")
    with c2:
        st.subheader("Prediction")
        st.write("• Agent-Level Forecasting")
        st.write("• Scene-Level Prediction")
    with c3:
        st.subheader("Planning")
        st.write("• Unified Cost Function")
        st.write("• End-to-End Training")
        st.write("• Minimize Error Propagation")

def ml_pipeline_page():
    st.title("End-to-End ML Pipeline")
    st.subheader("Multi-Task Learning Structure")

    st.write("1. Data Preprocessing & Augmentation")
    st.write("2. Shared Feature Extraction (CNN/Transformer)")
    st.write("3. Task-Specific Heads: Tracking, Occupancy, Forecasting, Planning")
    st.write("4. Query-Based Planning for Real-Time Decision Making")
    st.write("5. Joint Backpropagation with Weighted Loss")

    c1, c2 = st.columns(2)
    with c1:
        code = r"""
import torch
import torch.nn as nn

class UnifiedBackbone(nn.Module):
    def __init__(self):
        super(UnifiedBackbone, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

class PlanningHead(nn.Module):
    def __init__(self):
        super(PlanningHead, self).__init__()
        self.fc = nn.Linear(64*32*32, 128)
        self.out = nn.Linear(128, 10)  # e.g., 10-dim for trajectory
    def forward(self, features, sub_outputs):
        x = features.view(features.size(0), -1)
        x = self.fc(x)
        x = nn.functional.relu(x)
        # combine sub_outputs if needed
        return self.out(x)
        """
        st.code(code, language="python")

    with c2:
        image_pipeline = generate_placeholder_image("Data Flow & Module Interaction", (400, 300))
        st.image(image_pipeline)

def live_demo_page():
    st.title("Live Demo (Simulation)")

    st.write("Run a forward pass through a dummy model to simulate DriveVision's pipeline.")

    run_demo = st.button("Run Simulation")

    if run_demo:
        st.write("Running inference through dummy model...")
        start_time = time.time()
        out = model(fake_input)
        end_time = time.time()
        st.write("Output shape:", out.shape)
        st.write("Inference time (approx):", round(end_time - start_time, 4), "seconds")
        st.bar_chart(out.detach().numpy().flatten())

def data_viz_page():
    st.title("DriveVision Sensor Data Visualization")
    st.write("Below is a sample dataset from hypothetical sensor readings.")

    st.dataframe(dummy_data)

    st.line_chart(dummy_data.set_index("Time"))

    st.write("Correlation Matrix")
    corr = dummy_data[["Sensor1", "Sensor2", "Sensor3"]].corr()
    st.write(corr)

    st.write("Heatmap (Seaborn)")
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

# --------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
