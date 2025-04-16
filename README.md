# 🚢 Titanic Survival Prediction with K-Nearest Neighbors (KNN)

This project predicts passenger survival on the Titanic using a K-Nearest Neighbors (KNN) classifier with the **scikit-learn** library. It includes data preprocessing, feature engineering, hyperparameter tuning, model evaluation, and a confusion matrix visualization.

---

## 📁 Project Structure

- `main.py`: Run the full pipeline as a script.
- `titanic_survival_prediction.ipynb`: Interactive version of the pipeline with Jupyter Notebook.
- `titanic.csv`: Dataset (download from [Kaggle](https://www.kaggle.com/competitions/titanic/data)).

---

## ✅ Features

- **Data Cleaning**: Handles missing values and removes irrelevant columns.
- **Feature Engineering**: Adds `FamilySize`, `IsAlone`, `FareBin`, and `AgeBin`.
- **Model Training**: Uses **KNN** with **GridSearchCV** for tuning.
- **Evaluation**: Accuracy score + confusion matrix using Seaborn.
- **Visualization**: Confusion matrix heatmap.

---

## 📦 Dependencies

Make sure you have **Python 3.8+** and the following packages installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📅 Dataset

1. Download [`titanic.csv`](https://www.kaggle.com/competitions/titanic/data) from Kaggle.
2. Place it in the project directory (same folder as `main.py` or `titanic_survival_prediction.ipynb`).

---

## 🚀 How to Run

### 🔧 Option 1: Python Script

```bash
python main.py
```

**Expected Output:**

- Accuracy Score (e.g., `80.27%`)
- Confusion Matrix:
  ```
  [[115  19]
   [ 25  64]]
  ```
- Seaborn heatmap visualization

---

### 📓 Option 2: Jupyter Notebook

```bash
jupyter notebook
```

1. Open `titanic_survival_prediction.ipynb`
2. Run each cell in order

**Expected Output:**

- Same results as the script
- Interactive visualization in notebook

---

## 🧠 Code Workflow

1. **Imports**: Required libraries
2. **Data Loading**: Reads `titanic.csv`
3. **Preprocessing**:
   - Drop irrelevant columns (`PassengerId`, `Name`, `Ticket`, `Cabin`, `Embarked`)
   - Fill missing values (e.g., Age by Pclass median)
   - Encode `Sex` as numeric
   - Feature creation: `FamilySize`, `IsAlone`, `FareBin`, `AgeBin`
4. **Features and Target Split**
5. **Train/Test Split**: 75% train / 25% test
6. **Feature Scaling**: `MinMaxScaler`
7. **Model Training**: `GridSearchCV` for KNN tuning
8. **Evaluation**:
   - Accuracy score
   - Confusion matrix
9. **Visualization**: Confusion matrix heatmap with Seaborn

---

## 🗄️ Sample Output

```
Accuracy: 80.27%
Confusion Matrix:
[[115  19]
 [ 25  64]]
```

---

## 📈 Extensions

- Try other classifiers: `RandomForestClassifier`, `LogisticRegression`, etc.
- Add more features (e.g., extract titles from `Name` column)
- Replace seaborn with Plotly for interactive visualizations
- Generate synthetic data using `sklearn.datasets.make_blobs`

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 🙏 Acknowledgments

- [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)
- Built with ❤️ using:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

