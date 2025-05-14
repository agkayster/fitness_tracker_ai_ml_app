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

# to remove the NaN and have proper data for every row


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
