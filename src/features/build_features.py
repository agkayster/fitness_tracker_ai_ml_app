import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

# shows all the columns in the dataframe
# [:6] creates a subset or slice of the first 6 columns
# convert this to a "list" to use it later
predictor_columns = list(df.columns[:6])

# import Plot settings
plt.style.use("fiveThirtyEight")  # Use a predefined style
plt.rcParams["figure.figsize"] = (20, 5)  # Set default figure size
plt.rcParams["figure.dpi"] = 100  # Set default figure resolution
plt.rcParams["lines.linewidth"] = 2  # Set default line width


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

# using the interpolate function
for col in predictor_columns:
    df[col] = df[col].interpolate()

# this will show you there are no missing values
df.info()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 25]["acc_y"].plot()

# here we can calculate the duration of a set
# if you subtract 2 timestamps you get a time delta variable
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]

# we can call the seconds
duration.seconds

# use the loop to calculate the average duration of all sets
# s means each individual set number or id
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start
    # adds our duration column to the data table
    df.loc[(df["set"] == s), "duration"] = duration.seconds

# df shows our new table with the duration column
# we group by category and take the average duration of each category
duration_df = df.groupby(["category"])["duration"].mean()

# average duration per repetition
duration_df.iloc[0] / 5  # because it was repeated 5 times
duration_df.iloc[1] / 10  # medium set was repeated 10 times


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

# create a copy of the dataframe we would apply the lowpass filter to
df_lowpass = df.copy()

# creating a class instance of LowPassFilter class
Lowpass = LowPassFilter()

# this will result in 5, meaning 5 instances per second
# because the step size is 200 milliseconds
fs = 1000 / 200

cutoff = 1.2

# in the LowPassFilter class, we have the low_pass_filter function
df_lowpass = Lowpass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

# use a subset where we pick a set of 45 which is dead lift
subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

# we compare the butterworth filter plot against the raw data plot
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=[20, 10])
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# we apply this to all the columns using a loop
for col in predictor_columns:
    df_lowpass = Lowpass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
