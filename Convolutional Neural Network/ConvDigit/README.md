# 🧠 CNN Digit Recognition Project

This repository contains the **latest version of my Convolutional Neural Network (CNN)** for digit recognition.

It includes **two main scripts**:
- `AI-Digit.py`
- `ConvDigit.py`

---

## 📁 Project Structure

.  
├── Dataset/  
│ └── your_dataset.npz  
├── AI-Digit.py  
├── ConvDigit.py  
├── Add_data.py  
└── README.md  


---

## 🤖 AI-Digit.py

This script allows you to **create, train, test, and analyze a CNN model**.

### ▶️ How it works

1. **Launch the script**
2. Select the dataset you want to use  
   - The program searches for datasets in the `Dataset/` folder at the root of the repository.
   - Expected format: `.npz` file containing:
     - `X` → data
     - `y` → target labels
   - If your dataset does not match this format, use `Add_data.py`.

3. If you make a mistake during input, enter **`0`** to restart the process.

4. When asked whether to load the data, always answer **`yes`**,  
   **except** if you want to analyze kernels or biases of an existing model.

---

### 📊 Visualization

- Dataset information is printed in the terminal.
- All classes are displayed in a window (**press `SPACE` to skip**).
- A sample of the **training set** and **test set** is shown (**press `SPACE` to skip**).

---

### ⚙️ Available Modes

After initialization, choose one of the following modes:

1. **Create a new model** using custom parameters  
2. **Train a model** based on a previously saved model  
3. **Test a model** on a dataset sample  
4. **Analyze a model** (activations, kernels, and biases)

---

### 💾 Model Saving System

Models are saved automatically using the following structure:

Package_<DatasetName>/  
├── LogBook/  
│ └── model_logbook.csv  
└── Model/  
└── DM_<Accuracy><ConfidenceScore><Date>.pickle  


- **LogBook**: Contains all relevant information about each model.
- **Model**: Stores the trained CNN models.

---

## ✏️ ConvDigit.py

This script lets you **draw digits manually** and test them with a trained CNN model.

### ▶️ How it works

1. Select the dataset
2. Choose a trained model
3. Select the grid size:
   - `28` for **MNIST**
   - `8` for **Sklearn**
4. Choose the brush size:
   - Recommended:
     - `2` for MNIST
     - `1` for Sklearn
5. Choose whether to use pooling:
   - You can draw on a grid twice as large, then shrink it for more detail  
   - Recommendation:
     - **MNIST** → No
     - **Sklearn** → Yes

---

### 🖱️ Controls

- **Left click** → draw
- **Right click** → erase
- **SPACE** → validate drawing & display prediction
- **C** → clear the grid
- **ESC** or close window → exit program

---

## 🚀 Planned Improvements

- Support for **RGB images**
- **Faster training** process
