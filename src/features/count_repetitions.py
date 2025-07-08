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

plot_df = bench_df

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()


plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyro_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyro_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyro_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyro_r"].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

# specify the sampling frequency
# because there are 5 instances per second
fs = 1000 / 200

# initialize the LowPassFilter
LowPass = LowPassFilter()


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

# we look at the plots from a single set perspective
# below we have all single sets
bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]

# we then apply the lowpass filter to the single sets
# this gives us a messy pattern, not smooth
bench_set["acc_r"].plot()

# we need to use the lowpass filter to try and visualise something that resembles 5 repetitions
# bench_set["acc_r"] is our data table
# fs is our sampling frequency
# cutoff_frequency is 0.4
# order is 5
# column is "acc_r"
column = "acc_r"

# this would produce a new column called "acc_r_lowpass"
# the lowpass filter function gives us a smooth pattern
# because with lowpass filter, we cut off all the noise
LowPass.low_pass_filter(
    bench_set,
    col=column,
    sampling_frequency=fs,
    cutoff_frequency=0.4,
    order=5,
)[column + "_lowpass"].plot()

# the above would show us the 5 repetitions in the plot both in the minimum and maximum


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


# we now have to put everything in a function, so as to build dynamically
def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
    # we need a function to help us count the peaks and valleys (repetitions)
    # use the scipy.signal to achieve the counts
    # column = "acc_r"
    data = LowPass.low_pass_filter(
        dataset,
        col=column,
        sampling_frequency=fs,
        cutoff_frequency=cutoff,
        order=order,
    )
    # we need a set
    # bench_set["acc_r"].values this is a numpy array we pass into argrelextrema
    # data[column + "_lowpass"].values this is a numpy array we pass into argrelextrema
    # np.greater is the comparator we are using
    # we first get an array of indexes if we run the below
    # argrelextrema(bench_set["acc_r"].values, np.greater)
    # the argrelextrema function produces 5 values i.e. 5 repetitions
    argrelextrema(data[column + "_lowpass"].values, np.greater)

    # next step is to implement the visualisation
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    # this would identify the 5 peaks i.e. 5 maximum repetitions/count
    peaks = data.iloc[indexes]

    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    plt.title(f"{category} {exercise}:{len(peaks)} Reps")
    plt.show()

    return len(peaks)


# this calculates the total amount of reps or repetitions
count_reps(bench_set, cutoff=0.4)
count_reps(squat_set, cutoff=0.35)
count_reps(row_set, cutoff=0.65, column="gyro_x")
count_reps(ohp_set, cutoff=0.35)
count_reps(dead_set, cutoff=0.4)


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)

# this would give us the ground truth. Ensuring there are no mistakes in the data
rep_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index()

# adding a "reps_pred" meaning predicted reps column to the data table
rep_df["reps_pred"] = 0

# we now have to loop over everything in our count_reps function
# do our calculations and add it to the "rep_df" dataframe

for s in df["set"].unique():
    subset = df[df["set"] == s]

    column = "acc_y"
    cutoff = 0.4

    if subset["label"].iloc[0] == "squat":
        cutoff = 0.35

    if subset["label"].iloc[0] == "row":
        cutoff = 0.65
        column = "gyro_x"

    if subset["label"].iloc[0] == "ohp":
        cutoff = 0.35

    reps = count_reps(subset, cutoff=cutoff, column=column)

    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps

rep_df

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)

# create a quick plot to see where we went wrong
rep_df.groupby(["label", "category"])["reps", "reps_pred"].mean().plot.bar()