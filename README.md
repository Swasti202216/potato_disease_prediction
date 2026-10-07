# potato_disease_prediction

# 🥔 Potato Plant Disease Detection using Deep Learning

A Convolutional Neural Network (CNN) that identifies diseases in potato leaves from photographs. Given an image of a potato leaf, the model classifies it as one of three classes:

- Early Blight
- Late Blight
- Healthy

Early detection of these diseases helps farmers treat crops quickly, reduce losses and use fewer pesticides.
## 📌 Problem Statement

Early and late blight are among the most common and damaging potato diseases. Identifying them manually needs expert knowledge and is often slow or inaccurate. This project uses deep learning to automate the diagnosis from a simple leaf image.

## 📂 Dataset

- Source:[PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) (potato subset)
- Total images:2,152
- Classes (3): `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`
- Image size: resized to 256 × 256 × 3

| Split      | Share | Batches (size 32) |
|------------|-------|-------------------|
| Training   | 80%   | 54                |
| Validation | 10%   | 6                 |
| Testing    | 10%   | 8                 |

## 🧠 Model Architecture

A Sequential CNN built with TensorFlow/Keras:

1. Preprocessing layers: Resizing (256×256) and Rescaling (1/255)
2. Data augmentation: Random horizontal and vertical flip
3. Feature extraction: 6 × (Conv2D + MaxPooling2D) blocks — 32 filters in the first, 64 in the rest, all 3×3 kernels with ReLU
4. Classifier: Flatten → Dense(64, ReLU) → Dense(3, Softmax)

Training configuration

| Setting       | Value                              |
|---------------|------------------------------------|
| Optimizer     | Adam                               |
| Loss          | Sparse Categorical Crossentropy    |
| Metric        | Accuracy                           |
| Batch size    | 32                                 |
| Epochs        | 8                                  |

The input pipeline uses `tf.data` with caching, shuffling and prefetching for faster training.

## 📊 Results

| Metric                | Value                     |
|-----------------------|---------------------------|
| Training accuracy     | ~90.4%                    |
| Validation accuracy   | ~91.7%                    |
| Test accuracy         | **XX.XX%** *(update after re-running)* |


## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Install dependencies
   ```bash
   pip install tensorflow numpy matplotlib jupyter
   ```

3. Download the dataset
   Download the potato classes from PlantVillage and place them in a folder named `PlantVillage/` next to the notebook:
   ```
   PlantVillage/
   ├── Potato___Early_blight/
   ├── Potato___Late_blight/
   └── Potato___healthy/
   ```

4. Open and run the notebook
   ```bash
   jupyter notebook Training.ipynb
   ```


## 🔮 Future Improvements

- Fix the dataset split so train, validation and test sets stay fixed (use `reshuffle_each_iteration=False` and don't shuffle the test set)
- Add more augmentation (rotation, zoom, contrast) and regularization (Dropout, early stopping)
- Use transfer learning (e.g. MobileNetV2, EfficientNet) for higher accuracy
- Add a confusion matrix and per-class precision/recall
- Save the trained model and deploy it with FastAPI or Streamlit
- Build a mobile or web app so farmers can upload a leaf photo and get an instant diagnosis

## 🙏 Acknowledgements

- [PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) for the leaf images
- TensorFlow/Keras documentation and community

