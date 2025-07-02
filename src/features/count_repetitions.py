import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

# load data from "01_data_processed.pkl" file
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# remove rest period data from the dataset"
df = df[df["label"] != "rest"]

# we now calculate the sum of squares because there might be useful information as regards counting repetitions
# square all axis and add them together
acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyro_r = df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2

# get the square root of acc_r and gyro_r
df["acc_r"] = np.sqrt(acc_r)
df["gyro_r"] = np.sqrt(gyro_r)


# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

# we would split the data
# create 5 dataframes for each exercise
# we can study and reference them a lot easier

# define dataframe for bench press exercise
bench_df = df[df["label"] == "bench"]

# define dataframe for ohp press exercise
ohp_df = df[df["label"] == "ohp"]

# define dataframe for squat exercise
squat_df = df[df["label"] == "squat"]

# define dataframe for deadlift exercise
dead_df = df[df["label"] == "dead"]

# define dataframe for row exercise
row_df = df[df["label"] == "row"]




# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
