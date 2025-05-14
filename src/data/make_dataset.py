import pandas as pd

from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyro = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
len(files)


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = "../../data/raw/MetaMotion/"

f = files[0]

# we extract 3 variables from the filename
# 1. participant
# 2. label(exercise)
# 3. category of the set (eg. light or heavy)

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

df = pd.read_csv(f)

# we add extra columns to the dataframe
df["participant"] = participant
df["label"] = label
df["category"] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

# create accelerometer and gyroscope empty df
acc_df = pd.DataFrame()
gyro_df = pd.DataFrame()

# create a set. use this to increment the set number and create a unique identifier
acc_set = 1
gyro_set = 1

# build a loop to loop over all the files
for f in files:
    # for each file we extract the participant, label and category
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)

    # we add extra columns to the dataframe in each file
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Gyroscope" in f:
        df["set"] = gyro_set
        gyro_set += 1
        gyro_df = pd.concat([gyro_df, df])

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

# looking at the different datetimes
acc_df.info()

pd.to_datetime(df["epoch (ms)"], unit="ms")

gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")
acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")

# we remove other date and time columns
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyro_df["epoch (ms)"]
del gyro_df["time (01:00)"]
del gyro_df["elapsed (s)"]

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")

# define a function


def read_data_from_files(files):
    # create accelerometer and gyroscope empty df
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    # create a set. use this to increment the set number and create a unique identifier
    acc_set = 1
    gyro_set = 1

    # build a loop to loop over all the files
    for f in files:
        # for each file we extract the participant, label and category
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        # we add extra columns to the dataframe in each file
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df, df])

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")

    # we remove other date and time columns
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyro_df["epoch (ms)"]
    del gyro_df["time (01:00)"]
    del gyro_df["elapsed (s)"]

    return acc_df, gyro_df


acc_df, gyro_df = read_data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

# axis = 1 means we are merging column wise
# axis = 0 means we are merging row wise
# in order not to have duplicate columns
data_merged = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)

# rename columns
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "label",
    "category",
    "participant",
    "set",
]


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ 1/12
# Gyroscope:        25.000Hz 1/25

# here we are saying for all numerical values over a 200ms period we want to take the mean (average)
# for the "label", "category", "participant" and "set" we want to take the last value
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyro_x": "mean",
    "gyro_y": "mean",
    "gyro_z": "mean",
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last",
}

# for every 200ms we get as much information as possible for 1000 records
# [:1000] is used to limit the number of records to 1000
data_merged = data_merged[:1000].resample(rule="200ms").apply(sampling)

# split into days using a list comprehension method
# this is a list with data frames for each day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

# we do the resampling per day and merge it all together for a week
# this a list comprehesnsion way of saying, we loop over days and for each df in days we apply the resample rule with the following sample and we would drop any row where there is a null or no data
data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

data_resampled.info()

# change our "set" type to integer instead of float
data_resampled["set"] = data_resampled["set"].astype("int")


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

# Export data_resampled into our external data folder to intermi folder as a pickle file

# pickle files are ideal to use because they are smaller in size, faster to load and you do not bother about conversions when exporting or reading them again

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
