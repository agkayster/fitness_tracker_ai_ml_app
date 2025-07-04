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
# bench_set["acc_r"] is out data table
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

# we need a function to help us count the peaks and valleys (repetitions)
# use the scipy.signal to achieve the counts

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
