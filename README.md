# 🌸 Iris Flower Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Data Visualization](https://img.shields.io/badge/Data%20Visualization-Seaborn-green)
![Beginner Friendly](https://img.shields.io/badge/Project-Beginner%20Friendly-success)

## 📝 Project Overview
This project focuses on building a Machine Learning model to classify Iris flowers into three distinct species: **Setosa, Versicolor, and Virginica**. Based on the famous Fisher's Iris dataset, the model learns from the physical measurements of the flowers to accurately predict the correct species.

This is a fundamental classification problem that demonstrates data preprocessing, exploratory data analysis (EDA), model training, and performance evaluation.

## 📊 About the Dataset
The dataset consists of 150 samples of Iris flowers, with 50 samples from each of the three species. 
It contains the following **4 features** (measured in centimeters):
1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

**Target Variable:**
- `0`: Iris Setosa
- `1`: Iris Versicolor
- `2`: Iris Virginica

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn` (for pairplots and correlation heatmaps)
- **Machine Learning:** `scikit-learn`

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rahul-1419/Iris_Classification.git
   cd Iris_Classification
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. **Run the Code:**
   - If you are using a Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
     *(Open the `.ipynb` file to view the analysis and training process)*
   - If you are using a Python script:
     ```bash
     python main.py
     ```

## 📈 Results & Evaluation
The models were evaluated using **Accuracy Score, Precision, Recall, and Confusion Matrices**. 
- The dataset is highly linearly separable (especially the Setosa class), resulting in a high model accuracy of **~96% to 100%** depending on the algorithm and test-train split.

## 📂 Repository Structure
- `data/` - Contains the `iris.csv` dataset.
- `notebooks/` - Jupyter notebooks containing EDA and model training.
- `src/` or `main.py` - Main python scripts for the project.
- `README.md` - Project documentation.
