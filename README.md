Titanic Survival Prediction with K-Nearest Neighbors
Overview
This project predicts passenger survival on the Titanic using a K-Nearest Neighbors (KNN) classifier implemented with scikit-learn. The code, available as a Python script (main.py) and a Jupyter notebook (scikit-learn.ipynb), performs data preprocessing, feature engineering, hyperparameter tuning, model evaluation, and visualization of a confusion matrix.
Features

Data Cleaning: Handles missing values and removes irrelevant columns.
Feature Engineering: Creates features like FamilySize, IsAlone, FareBin, and AgeBin.
Model Training: Uses KNN with hyperparameter tuning via GridSearchCV.
Evaluation: Computes accuracy and visualizes a confusion matrix using seaborn.
Dataset: Titanic dataset (titanic.csv), available from Kaggle.

Prerequisites

Python: Version 3.8 or higher.
Dependencies:
pandas
numpy
scikit-learn
matplotlib
seaborn


Dataset: Download titanic.csv from Kaggle and place it in the project directory.

Installation

Clone or Download the Project:

Clone this repository or download main.py and scikit-learn.ipynb.


Install Dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn


Set Up the Dataset:

Place titanic.csv in the same directory as main.py or scikit-learn.ipynb.



Usage
The project can be run as a Python script (main.py) or in a Jupyter notebook (scikit-learn.ipynb).
Running the Python Script

Ensure titanic.csv is in the same directory as main.py.

Run the script:
python main.py


Expected Output:

Dataset information and missing value counts.

Model accuracy and confusion matrix, e.g.:
Accuracy: 80.27%
Confusion Matrix:
[[115  19]
 [ 25  64]]


A seaborn heatmap displaying the confusion matrix.




Running the Jupyter Notebook

Open Jupyter Notebook:
jupyter notebook


Open scikit-learn.ipynb.

Ensure titanic.csv is in the same directory.

Run each cell sequentially to:

Load and preprocess the data.
Train and tune the KNN model.
Evaluate the model and visualize the confusion matrix.


Expected Output: Same as the script, with an interactive confusion matrix plot.


Code Structure
The code is organized into logical sections:

Imports: Libraries for data handling, modeling, and visualization.
Data Loading: Loads titanic.csv and displays basic info.
Preprocessing:
Drops columns: PassengerId, Name, Ticket, Cabin, Embarked.
Fills missing Age values based on Pclass median.
Converts Sex to numeric.
Creates features: FamilySize, IsAlone, FareBin, AgeBin.


Feature and Target Preparation: Splits data into features (X) and target (y).
Data Splitting: Creates training (75%) and testing (25%) sets.
Scaling: Applies MinMaxScaler to features.
Hyperparameter Tuning: Uses GridSearchCV to optimize KNN parameters.
Evaluation: Computes accuracy and confusion matrix.
Visualization: Plots the confusion matrix using seaborn.

Notebook Version
The scikit-learn.ipynb notebook splits the code into cells for interactivity:

Imports
Data loading and inspection
Preprocessing functions
Data preprocessing
Feature/target preparation
Data splitting
Feature scaling
Hyperparameter tuning
Model training
Model evaluation
Results printing
Confusion matrix plotting

Example Output
Running main.py or scikit-learn.ipynb produces:

Console Output:
Accuracy: 80.27%
Confusion Matrix:
[[115  19]
 [ 25  64]]


Visualization: A seaborn heatmap of the confusion matrix, showing true vs. predicted survival labels.


Notes

Dataset: The code assumes titanic.csv is in the working directory. Update the file path if needed (e.g., pd.read_csv("/path/to/titanic.csv")).
Performance: The KNN model achieves ~80.27% accuracy, as shown in the example output.
Visualization: The confusion matrix uses seaborn. To use Plotly for interactive plots, modify the plot_model function.
Extending the Project:
Experiment with other classifiers (e.g., RandomForestClassifier).
Add features (e.g., extract titles from Name).
Generate synthetic data with sklearn.datasets.make_blobs for a generative AI approach.


License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset provided by Kaggle.
Built with scikit-learn, pandas, and seaborn.

