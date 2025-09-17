# Airbus Ship Detection

## 📌 Project Overview
This repository contains the solution developed for the **Airbus Ship Detection Challenge** on Kaggle.  
The goal of the challenge is to detect ships in high-resolution satellite imagery, a task relevant for **surveillance, logistics, and environmental monitoring**.  

Ship detection is challenging because ships are often **small, sparse, or occluded**, requiring models that balance **precision, generalization, and computational efficiency**.  
This project tackles the problem as a **binary semantic segmentation** task using a **custom CNN with a U-Net backbone** and **Squeeze-and-Excitation (SE) blocks**.  

---

## 📂 Dataset
- Source: [Airbus Ship Detection Dataset](https://www.kaggle.com/competitions/airbus-ship-detection/data)  
- Contains **~208k high-resolution satellite images** (768×768).  
- Issues: **class imbalance**, **mask encoding (RLE)**, and **high visual confusion** due to cluttered backgrounds.  

### Preprocessing & Construction
- Each image is divided into **9 patches (256×256)**.  
- Balanced dataset:  
  - **Training:** 20k images (80% ship patches, 20% empty)  
  - **Validation:** 2k images (50/50 ship-empty)  
  - **Test:** 500 images (50/50 ship-empty)  
- **Augmentations:**  
  - Geometric: flips, rotations  
  - Color: brightness/contrast changes, normalization  

---

## ⚙️ Model Architecture
The network follows a **U-Net** design with added **Squeeze-and-Excitation (SE) blocks** for channel reweighting.  

- **Encoder:** Convolutional blocks with Group Normalization, SiLU activations, Dropout, and SE blocks.  
- **Decoder:** Transposed convolutions with skip connections.  
- **Output Layer:** Sigmoid activation for binary mask prediction.  

**Loss Function (Custom Combined Loss):**
- **Binary Cross-Entropy (BCE):** pixel-level classification  
- **Dice Loss:** effective for imbalanced data and small object detection  
- **Final Loss:** weighted sum of BCE + Dice  

**Hyperparameters:**
- Epochs: 20  
- Learning Rate: 0.001 (ReduceLROnPlateau scheduler)  
- Optimizer: AdamW (weight decay = 0.0001)  

---

## 📊 Results

### Training Performance
- **Train Dice ≈ 0.60**  
- **Validation Dice ≈ 0.59**  
- No overfitting, smooth convergence (~13 min/epoch, ~5h total).  

### External Test Evaluation
- **Mean Patch Dice:** 78.88%  
- **Mean Ship Dice:** 65.76%  
- **Mean Empty Dice:** 92.00%  
- **Macro Dice:** 78.88%  

These results demonstrate a **computationally efficient and modular baseline** for ship detection.  

---

## 🔬 Comparison with SOTA
- **Cascaded Mask R-CNN**: stronger detection, but higher computational cost.  
- **Frequency domain approaches (Wavelet transforms, Faster R-CNN, etc.)**: superior segmentation accuracy in recent research.  

Our approach is a **lightweight alternative** that achieves reasonable performance with limited resources.  

---

## 🚀 Future Directions
- Scale training with larger datasets and stronger GPUs.  
- Explore **region proposal** and **frequency-domain architectures**.  
- Reformulate as a **two-stage pipeline (classification + detection)**.  

---

## 🖥️ How to Run the Project

1. **Install uv** (Python package manager):  
   👉 [Installation guide](https://astral.sh/blog/uv/)  

2. **Create and activate a virtual environment** with uv.  

3. **Install dependencies:**  
   - Preferred: use the `pyproject.toml`  
   - Alternative: `requirements.txt` (if available)  

4. **Download the dataset** from Kaggle:  
   [Airbus Ship Detection Dataset](https://www.kaggle.com/competitions/airbus-ship-detection/data)  

   After extraction, your project root should contain:  
   - train_v2/
   - test_v2/
   - train_ship_segmentations_v2.csv

5. **Run the notebook:**  
Open `final_main.ipynb` in Jupyter Notebook and execute all cells (preprocessing → training → evaluation).  

6. **Launch the dashboard:**  
```bash
streamlit run dashboard.py

