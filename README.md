# Visual Question Answering (VQA) using BLIP and PyTorch

This project implements a Visual Question Answering (VQA) system that combines image and text understanding to answer questions based on visual input. It leverages the **BLIP (Bootstrapped Language Image Pretraining)** model from Hugging Face and is built using **PyTorch**.

## Project Overview

Visual Question Answering (VQA) is a challenging task at the intersection of computer vision and natural language processing. The goal is to create an AI model that can understand an image and answer natural language questions about it.

In this project:
- We use a pre-trained BLIP model to handle multi-modal data (images + questions).
- A custom dataset class and dataloader are built to load image-question-answer triples.
- GPU acceleration is used for performance efficiency.
- Outputs are visualized to interpret model predictions.

## Directory Structure

aods-project/

├── dataset/ # Image and question files

├── outputs/ # Visualized results and sample predictions

├── aods-project.ipynb # Main Jupyter Notebook


├── README.md # Project documentation

└── requirements.txt # Python dependencies


## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/aods-project.git
cd aods-project'''

## 2. Create a virtual environment (optional)
python -m venv vqa-env
source vqa-env/bin/activate  # On Windows: vqa-env\Scripts\activate

## 3. Install dependencies
pip install -r requirements.txt
If requirements.txt is missing, install the main libraries manually:
pip install torch torchvision transformers pandas matplotlib pillow

## 4. Run the Notebook
Launch the Jupyter Notebook and run the aods-project.ipynb step-by-step to:
Load the model
Process the image-question pairs
Visualize the answers

## Model: BLIP
BLIP (Bootstrapped Language Image Pretraining) is a vision-language model designed for tasks like image captioning, VQA, and image-text retrieval. We use the BLIP Question Answering variant from Hugging Face.
Model Used:
Salesforce/blip-vqa-base from Hugging Face

## Sample Output
Image: 🖼️
Question: "What is the man doing?"
Model Answer: "Playing guitar"
(Visualizations are available in the outputs/ folder.)

## Features
Transformer-based vision-language model
Supports any image and natural language question
GPU-compatible for faster processing
Clean and modular code structure
