import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

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
set_df = df[df["set"] == 1]

# plots our accelerometer data for the x-axis as against time on the y-axis
plt.plot(set_df["acc_y"])

# this is using the reset_index method to reset the index of the DataFrame
# and we use the index as our horizontal axis, which tells us that there are more than 80 samples
plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

# this will give us the unique labels in the label column
df["label"].unique()

# we are looping over the unique labels in the label column
# and for each label we are creating a subset of the data frame
for label in df["label"].unique():
    subset = df[df["label"] == label]
    # display dataframes within a loop
    # display(subset.head(2))
    # we now plot the subset
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# here we are visualizing the whole set of each exercise in one figure

# here we want to view details of the first 100 records
# for every subset, we plot the first 100 samples
for label in df["label"].unique():
    subset = df[df["label"] == label]
    # display dataframes within a loop
    # display(subset.head(2))
    # we now plot the subset
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
# here we are setting the font size and fig size for the plots
mpl.style.use("seaborn-v0_8-deep")

mpl.rcParams["figure.figsize"] = (10, 5)
mpl.rcParams["figure.dpi"] = 100



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
