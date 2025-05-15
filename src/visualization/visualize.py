import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

# read our pickle file
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
# create a data frame for a specific set
# create a subset of the original data frame using the set number from the set column
set_df = df[df["set"] == 64]

# plots our accelerometer data for the x-axis as against time on the y-axis
plt.plot(set_df["acc_x"])

# this is using the reset_index method to reset the index of the DataFrame
# and we use the index as our horizontal axis, which tells us that there are more than 80 samples
plt.plot(set_df["acc_x"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
