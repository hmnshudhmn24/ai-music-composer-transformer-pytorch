
---

### 🧠 `music_transformer.py`

```python
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from music21 import converter, instrument, note, chord, stream

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Hyperparameters
SEQ_LENGTH = 50
BATCH_SIZE = 64
EPOCHS = 50
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
NUM_LAYERS = 2
LEARNING_RATE = 0.001

class MusicDataset(Dataset):
    def __init__(self, file_list, seq_length):
        self.notes = []
        for file in file_list:
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts:  # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    self.notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    self.notes.append('.'.join(str(n) for n in element.normalOrder))
        # Create note to int and int to note mappings
        self.pitchnames = sorted(set(self.notes))
        self.note_to_int = dict((note, number) for number, note in enumerate(self.pitchnames))
        self.int_to_note = dict((number, note) for number, note in enumerate(self.pitchnames))
        # Prepare sequences
        self.network_input = []
        self.network_output = []
        for i in range(0, len(self.notes) - seq_length):
            seq_in = self.notes[i:i + seq_length]
            seq_out = self.notes[i + seq_length]
            self.network_input.append([self.note_to_int[char] for char in seq_in])
            self.network_output.append(self.note_to_int[seq_out])
        self.n_vocab = len(self.pitchnames)

    def __len__(self):
        return len(self.network_input)

    def __getitem__(self, idx):
        return torch.tensor(self.network_input[idx], dtype=torch.long), torch.tensor(self.network_output[idx], dtype=torch.long)

class TransformerModel(nn.Module):
    def __init__(self, n_vocab, embedding_dim, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead=4, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(embedding_dim, n_vocab)

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(EMBEDDING_DIM)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

def train_model(model, dataset, device, model_path):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, dataset.n_vocab)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader)}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def generate_music(model, dataset, device, output_dir, sequence_length=SEQ_LENGTH, generate_length=500):
    model.eval()
    int_to_note = dataset.int_to_note
    note_to_int = dataset.note_to_int
    start = np.random.randint(0, len(dataset.network_input)-1)
    pattern = dataset.network_input[start]
    prediction_output = []
    with torch.no_grad():
        for note_index in range(generate_length):
            input_seq = torch.tensor(pattern, dtype=torch.long).unsqueeze(0).to(device)
            prediction = model(input_seq)
            prediction = prediction[:, -1, :]
            _, top_i = prediction.topk(1)
            next_index = top_i[0][0].item()
            prediction_output.append(int_to_note[next_index])
            pattern.append(next_index)
            pattern = pattern[1:]
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "output.mid")
    midi_stream.write('midi', fp=output_path)
    print(f"Generated music saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="AI Music Composer with Transformer")
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True, help='Mode: train or generate')
    parser.add_argument('--data_dir', type=str, help='Directory containing MIDI files')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save/load the model')
    parser.add_argument('--output_dir', type=str, default='generated_music', help='Directory to save generated music')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        if not args.data_dir:
            print("Please provide --data_dir for training.")
            return
        midi_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.mid')]
        dataset = MusicDataset(midi_files, SEQ_LENGTH)
        model = TransformerModel(dataset.n_vocab, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
        train_model(model, dataset, device, args.model_path)
    elif args.mode == 'generate':
        if not os.path.exists(args.model_path):
            print(f"Model file {args.model_path} not found.")
            return
        if not args.data_dir:
            print("Please provide --data_dir for dataset to initialize.")
            return
        midi_files
::contentReference[oaicite:0]{index=0}
