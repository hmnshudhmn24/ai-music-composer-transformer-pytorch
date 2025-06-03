# 🎵 AI Music Composer with Transformer

## 🧠 Project Description

This project leverages a Transformer-based neural network to generate MIDI music sequences. Trained on a dataset of classical music, the model learns to compose original melodies, demonstrating the fusion of deep learning and music theory.

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ai-music-composer-transformer.git
cd ai-music-composer-transformer
```

### 2️⃣ Install Required Libraries

Ensure you have Python 3.7 or higher installed. Then, install the dependencies:

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare the Dataset

Place your MIDI files in a directory, for example, `./midi_data/`. Ensure the directory contains `.mid` files.

### 4️⃣ Train the Model

Run the training script:

```bash
python music_transformer.py --mode train --data_dir ./midi_data/ --model_path ./model.pth
```

### 5️⃣ Generate Music

After training, generate new music sequences:

```bash
python music_transformer.py --mode generate --model_path ./model.pth --output_dir ./generated_music/
```

## 🧰 Tech Stack

- **PyTorch** – Deep learning framework for model implementation.
- **music21** – Library for parsing and handling MIDI files.
- **NumPy** – Numerical operations.

## 🌟 Key Features

- 🎼 Transformer-based architecture for sequence modeling.
- 🎹 Parses MIDI files into note sequences for training.
- 🎶 Generates new MIDI files based on learned patterns.
