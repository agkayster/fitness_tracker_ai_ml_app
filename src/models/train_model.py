import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# load our data from previous pickle file
df = pd.read_pickle("../../data/interim/03_data_features.pkl")


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

# remove some columns from the dataset that we would not be using
df_train = df.drop(["participant", "category", "set"], axis=1)

# split the data into the X and Y variables of the training set
# label points to our exercise labels e.g. "squat", "lunges", etc.
# X variable is taking all columns except the label column from the training set
X = df_train.drop("label", axis=1)

# y variable is taking the "label" column from the training set
y = df_train["label"]

# use "train_test_split" to split the data into training and test sets
# this takes control of the stochastic process that happens when splitting the data
# we set the test size to 25% of the data and the random state to 42 for reproducibility
# 75% of the data will be used for training and 25% for testing
# stratify=y ensures that the distribution of labels is maintained in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# create a quick plot to see the stratify in action
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="blue", label="Total", alpha=0.7
)
y_train.value_counts().plot(
    kind="bar", ax=ax, color="dodgerBlue", label="Train", alpha=0.7
)
y_test.value_counts().plot(
    kind="bar", ax=ax, color="royalBlue", label="Test", alpha=0.7
)
plt.legend()
plt.show()

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

# split the features into subsets based on the type of features

# original features in the dataset
basic_features = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z    "]

# square features
square_features = ["acc_r", "gyro_r"]

# principal component features
pca_features = ["pca_1", "pca_2", "pca_3"]

# time features
time_features = [f for f in df_train.columns if "_temp_" in f]

# freq features
freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]

# cluster features
cluster_features = ["cluster"]

print("Basic features:", len(basic_features))
print("Square features:", len(square_features))
print("PCA features:", len(pca_features))
print("Time features:", len(time_features))
print("Freq features:", len(freq_features))
print("Cluster features:", len(cluster_features))

# df_train.columns[30:]

# we now create 4 different feature sets
feature_set_1 = list(set(basic_features))

# set is a data type in python, used in order to avoid duplicates and create a unique collection of items
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------


# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
