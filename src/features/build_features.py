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

df.info()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------


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
