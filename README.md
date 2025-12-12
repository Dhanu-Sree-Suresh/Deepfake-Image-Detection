# DeepSense - Deepfake Detection

Multi-modal deepfake detection using deep learning and iris biometric analysis.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Training

```bash
python train.py
```

## Project Structure

```
DeepSense/
├── app.py              # Streamlit UI
├── train.py            # Training script
├── detector.py         # Heuristic detector
├── src/
│   ├── model.py        # Neural network
│   └── preprocessing.py
├── data/
│   ├── real/           # Real images
│   └── fake/           # Fake images
└── checkpoints/        # Saved models
```
