import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

# import the data as usual
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# get all the columns that have numerical values
# we use list, to turn the variable into a list/array
outlier_columns = list(df.columns[:6])

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------
# initiate our styling
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# here we create a boxplot using methods from the pandas library
df[["acc_y", "label"]].boxplot(by="label", figsize=(20, 10), fontsize=14)
df[["gyro_y", "label"]].boxplot(by="label", figsize=(20, 10), fontsize=14)

# we pass in the "outlier_columns" below
# this is for accelerometer data
df[outlier_columns[:3] + ["label"]].boxplot(
    by="label", figsize=(20, 10), fontsize=14, layout=(1, 3)
)

# we pass in the outlier columns below
# this is for gyroscope data
df[outlier_columns[3:] + ["label"]].boxplot(
    by="label", figsize=(20, 10), fontsize=14, layout=(1, 3)
)


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Insert IQR function


# Plot a single column


# Loop over all columns


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution


# Insert Chauvenet's function


# Loop over all columns


# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function


# Loop over all columns


# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column


# Create a loop

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
