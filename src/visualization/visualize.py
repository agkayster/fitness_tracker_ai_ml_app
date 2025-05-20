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

# here we want to spot any differences between the medium and heavy sets of any particular exercise
# here we define another subset
# use "query" method from pandas to create subsets

category_df = (
    df.query("label == 'squat'").query("participant == 'A'").reset_index(drop=True)
)

# now to comapre the medium and the heavy sets for the squat of participant A
# we create a grouped plot using "groupby" method

# to apply styling and make the labels show on the plots apart from the legend
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

# here we want to compare the bench exercise of all participants
# query the dataframe using a label of "bench" and sort the values by participant (alphabetical order)
# reset index to look at the samples not time intervals
participant_df = (
    df.query("label == 'bench'").sort_values("participant").reset_index(drop=True)
)

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

# we are plotting the x, y and z axis for the accelerometer data and gyroscope data
# we used the f-string here for querying for a variable, this would be done dynamically instead of statically
label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

# now we create a figure for our axis and plots
fig, ax = plt.subplots()
# we are creating a subset with multiple columns
# if you use single brackets, it will return a pandas series
# if you use double brackets, it will return a dataframe
# a series does not take multiple columns
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels = df["label"].unique()
participants = df["participant"].unique()

# this is for the accelerometer data
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )

        if len(all_axis_df) > 0:
            # now we create a figure for our axis and plots
            fig, ax = plt.subplots()
            # we are creating a subset with multiple columns
            # if you use single brackets, it will return a pandas series
            # if you use double brackets, it will return a dataframe
            # a series does not take multiple columns
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} - ({participant})".title())
            plt.legend()


# this is for the gyroscope data
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )

        if len(all_axis_df) > 0:
            # now we create a figure for our axis and plots
            fig, ax = plt.subplots()
            # we are creating a subset with multiple columns
            # if you use single brackets, it will return a pandas series
            # if you use double brackets, it will return a dataframe
            # a series does not take multiple columns
            all_axis_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax)
            ax.set_ylabel("gyro_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} - ({participant})".title())
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
