import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


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

df_lowpass

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

# create a df that we would apply PCA to
df_pca = df_lowpass.copy()

# create a new instance of the class PCA
PCA = PrincipalComponentAnalysis()

# should give a list of 6 values because there are 6 columns in total
# PCA is a dimensionality reduction method
# we move from alot of columns to less columns
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# look at methods to determine the optimal amount of principal components
# we use the ELBOW technique for this

# example of an ELBOW
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

# applying the PCA
# this will summarize our 6 columns (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) into 3 columns pca_1, pca_2, pca_3
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# we need to visualize our pca columns to get an understanding of what we have done
# for this we create a subset
subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

df_pca


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

# copy the dataframe from above
df_squared = df_pca.copy()

# square all axis and add them together
acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyro_r = (
    df_squared["gyro_x"] ** 2 + df_squared["gyro_y"] ** 2 + df_squared["gyro_z"] ** 2
)

# get the square root of acc_r and gyro_r
df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyro_r"] = np.sqrt(gyro_r)

# visualize the square root above
subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyro_r"]].plot(subplots=True)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

# create a copy of the last dataframe and assign it to a new dataframe
df_temporal = df_squared.copy()

# using the NumericalAbstraction class
# this initiates a new class instance
# by initializing the class below, we can use its methods/functions inside the class
NumAbs = NumericalAbstraction()

# add the new columns to the predictor columns
predictor_columns = predictor_columns + ["acc_r", "gyro_r"]

# Now we determine the window size
# this captures how many values we want to look back for each of those values
# finding the value of the window size is a bit of a trial and error process
# 200 is our step size, which is 200 milliseconds
# to get a window size of 1 second, we need 5 steps or 5 instances
# our window size is 5 because we want to look back 1 second
ws = int(1000 / 200)

# Loop through the predictor columns and compute the mean and standard deviation
for col in predictor_columns:
    # calculate the mean and standard deviation for each column
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")


df_temporal

# here we create a subset by splitting the above loop
# we then make a subset based on the set
# Compute the values of each of the individual sets
df_temporal_list = []

# loop over the unique sets in the dataframe
for s in df_temporal["set"].unique():
    # create a subset for each set
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        # calculate the mean and standard deviation for each column
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

# Concatenate the list of dataframes into a single dataframe
# we then overide the old df_temporal with the new one
df_temporal = pd.concat(df_temporal_list)

# by doing the above we have countered the spill over from the previous set into the next set
df_temporal.info()

# we check a typical subset
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]]

# we take a plot of the above subset
# this is for accelerometer
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
# this is for gyroscope
subset[["gyro_y", "gyro_y_temp_mean_ws_5", "gyro_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

# create a copy of the previous dataframe
# here we need to reset the index
df_freq = df_temporal.copy().reset_index()

# using the FourierTransformation class
# this initiates a new class instance
# by initializing the class below, we can use its methods/functions inside the class
FreqAbs = FourierTransformation()

# here we define the sampling rate
# timing interval
fs = int(1000 / 200)

# here we define the window size
# 2800 is average time for a set
ws = int(2800 / 200)

# we first apply the FourierTransformation to one column
# we take "df_freq" as our data table here
# ["acc_y"] is the one column we are using first
# the columns ["acc_y"] must be in a list ([])
df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

# to see all columns created
df_freq.columns

# to visualize all our data
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot(subplots=True)

# here we create a subset by splitting the above loop
# we then make a subset based on the set
# Compute the values of each of the individual sets
df_freq_list = []

# loop over the unique sets in the dataframe to apply FourierTransformation
for s in df_freq["set"].unique():
    print(f"Applying FourierTransformation to set {s}")
    # create a subset for each set
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)


# add the df_freq_list to the dataframe
pd.concat(df_freq_list)

# we store this in the dataframe
# use "set_index" to drop the dicreet index and use "epoch (ms)"
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# start by taking the latest dataframe
# this will drop rows with NaN values/missing values
df_freq = df_freq.dropna()

# to deal with overlapping windows, we get rid of some part of the data
# an allowance of 50% overlap is common, we get rid of 50% of the data
# do this by skipping every second row
# results in alot of data loss but pay off in the long run
# models are less prone to overfitting

# this means all rows and all columns using iloc
# df_freq.iloc[:,:]

# every column but the first two rows using iloc
# df_freq.iloc[:2]

# get every second row using iloc
df_freq = df_freq.iloc[::2, :]


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

# create a copy of the last dataframe
df_cluster = df_freq.copy()

# here we would be clustering based on the accelerometer data
cluster_columns = ["acc_x", "acc_y", "acc_z"]

# determine the amount of K we want to use
k_values = range(2, 10)

# store the inertias in a list
# we would first define as an empty list
inertias = []

# now we loop over the dataframe and create our clusters
for k in k_values:
    # create a subset based on the cluster columns
    subset = df_cluster[cluster_columns]
    # define our KMeans model
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    # specify the cluster labels
    # fit_predict is used to instantly train the model and make predictions
    # this would input the subset and create the cluster labels
    # for each of the rows in the subset, it would assign the row to a cluster
    # On the first iteration of the loop, k would be 2, a row can be assigned to cluster 1 or cluster 2
    # so it would be value 1 or value 2
    # On the second iteration of the loop, k would be 3, a row can be assigned to cluster 1, cluster 2 or cluster 3
    # and it would do that all the way until k is 10
    cluster_labels = kmeans.fit_predict(subset)

    # we store the inertias
    # we used the inertias to plot the on the graph
    # use the Elbow method to determine the optimal amount of k
    inertias.append(kmeans.inertia_)

# view the inertias list
inertias

# use the list to create a plot and visually inspect whether we can create an Elbow
# example of an ELBOW
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("K")
plt.ylabel("Sum of squared distances (inertia)")
plt.show()


kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# call df_cluster
df_cluster

# to visualize the clusters, we can use a scatter plot
# plot clusters in 3D
# projection="3d" is from matplotlib library
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

# looping over the dataframe and splitting it by cluster
for c in df_cluster["cluster"].unique():
    # loop over all the unique clusters
    # create a subset of the dataframe for each cluster
    subset = df_cluster[df_cluster["cluster"] == c]
    # create a scatter plot for each cluster
    # since projection="3d", we get a 3D plot with 3 different axis
    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        label=f"Cluster {c}",
        alpha=0.5,
    )
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# we need to compare above cluster to the same cluster but splitting by labels

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for label in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == label]
    ax.scatter(
        subset["acc_x"],
        subset["acc_y"],
        subset["acc_z"],
        label=f"Label {label}",
        alpha=0.5,
    )
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

# we export the final dataframe to a pickle file
df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
