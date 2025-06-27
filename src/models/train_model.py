import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# load our data from previous pickle file
df = pd.read_pickle("../../data/interim/03_data_features.pkl")


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

# remove some columns from the dataset that we would not be using
df_train = df.drop(["participant", "category", "set"], axis=1)

# split the data into the X and Y variables of the training set
# label points to our exercise labels e.g. "squat", "lunges", etc.
# X variable is taking all columns except the label column from the training set
X = df_train.drop("label", axis=1)

# y variable is taking the "label" column from the training set
y = df_train["label"]

# use "train_test_split" to split the data into training and test sets
# this takes control of the stochastic process that happens when splitting the data
# we set the test size to 25% of the data and the random state to 42 for reproducibility
# 75% of the data will be used for training and 25% for testing
# stratify=y ensures that the distribution of labels is maintained in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# create a quick plot to see the stratify in action
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="blue", label="Total", alpha=0.7
)
y_train.value_counts().plot(
    kind="bar", ax=ax, color="dodgerBlue", label="Train", alpha=0.7
)
y_test.value_counts().plot(
    kind="bar", ax=ax, color="royalBlue", label="Test", alpha=0.7
)
plt.legend()
plt.show()

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

# split the features into subsets based on the type of features

# original features in the dataset
basic_features = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# square features
square_features = ["acc_r", "gyro_r"]

# principal component features
pca_features = ["pca_1", "pca_2", "pca_3"]

# time features
time_features = [f for f in df_train.columns if "_temp_" in f]

# freq features
freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]

# cluster features
cluster_features = ["cluster"]

print("Basic features:", len(basic_features))
print("Square features:", len(square_features))
print("PCA features:", len(pca_features))
print("Time features:", len(time_features))
print("Freq features:", len(freq_features))
print("Cluster features:", len(cluster_features))

# df_train.columns[30:]

# we now create 4 different feature sets
feature_set_1 = list(set(basic_features))

# set is a data type in python, used in order to avoid duplicates and create a unique collection of items
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
learner = ClassificationAlgorithms()

max_features = 10  # maximum number of features to select

# in order to fire up the selection, we need to see everythng that it outputs
# this script will loop over all the individual columns which is 117 in total in the dataframe and it will train a decision tree/model 117 times and then it gets the best parameter
# after that, it will loop over the 116 left over columns apart from the best performing and do the training again
# it will run from 0 to 9, because we set the max_features to 10
# after it runs, we highlight the selected features, ordered features and ordered scores.
# selected features and ordered features are the same so we ignore one
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

# best practice
# you could get different results if you run the forward selection multiple times
selected_features = [
    "acc_z_freq_0.0_Hz_ws_14",
    "acc_x_freq_0.0_Hz_ws_14",
    "duration",
    "acc_y_temp_mean_ws_5",
    "gyro_r_temp_mean_ws_5",
    "gyro_r_freq_1.429_Hz_ws_14",
    "acc_z_freq_weighted",
    "gyro_r_max_freq",
    "gyro_z_temp_std_ws_5",
    "gyro_x_freq_1.071_Hz_ws_14",
]

ordered_features = [
    "acc_z_freq_0.0_Hz_ws_14",
    "acc_x_freq_0.0_Hz_ws_14",
    "duration",
    "acc_y_temp_mean_ws_5",
    "gyro_r_temp_mean_ws_5",
    "gyro_r_freq_1.429_Hz_ws_14",
    "acc_z_freq_weighted",
    "gyro_r_max_freq",
    "gyro_z_temp_std_ws_5",
    "gyro_x_freq_1.071_Hz_ws_14",
]

ordered_scores = [
    0.885556704584626,
    0.9889693209238194,
    0.9989658738366081,
    0.9993105825577387,
    0.9996552912788693,
    0.9996552912788693,
    0.9996552912788693,
    0.9996552912788693,
    0.9996552912788693,
    0.9996552912788693,
]


# we plot the chart to visualize the results
# plot shows the total number of features selected via forward selection features and the accuracy on the training data
# the acc_z_freq_0.0_Hz_ws_14 gives us an 0.885556704584626 % accuracy (best feature)
# the acc_x_freq_0.0_Hz_ws_14 is the second best feature gives us close to 99% accuracy (0.9889693209238194)
#
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()


# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

# possible feature sets
possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features,
]

feature_names = [
    "Feature Set 1",
    "Feature Set 2",
    "Feature Set 3",
    "Feature Set 4",
    "Selected Features",
]

# define the iterations variable
# start with 1 iteration to get a sense of how everything works
iterations = 1

# define a score dataframe to store the results
score_df = pd.DataFrame()


# Grid search code

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------

# now we look at our score dataframe
score_df.sort_values(by="accuracy", ascending=False)


# create a bar plot to see the effect of the models and feature sets
plt.figure(figsize=(10, 10))
sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()

# from the bar plot "selected features" and "Feature Set 4" are the best performing feature sets
# "selected features" is the best performing feature set with the highest accuracy
# "Feature Set 4" is the second best performing feature set with a slightly lower accuracy
# "Selected Features" and "Feature Set 4" have 99% accuracy
# the best performing model i see is the "Decision Tree" with 99.9% accuracy


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

# the best performing model is the "Decision Tree"
# here we are calling the learning class with decistion tree and we would train using the "X_train" and "y_train" data and validate using the "X_test" data
# we would also perform a grid search
# we pass in the "Selected Features" to the decision tree model
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.decision_tree(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
)

# use the output variables to calculate the accuracy score
# "class_test_y" is our prediction
# use our "y_test" to make an accuracy score
# "y_test" is the original labels for the test set
accuracy = accuracy_score(y_test, class_test_y)
print("Accuracy of the best model (Decision Tree):", accuracy)

# we set up our confusion matrix to see how well the model performed
# use the probabilities
# this shows all our labels
classes = class_test_prob_y.columns

# define a confusion matrix with variable "cm"
# "cm" is an array/list
cm = confusion_matrix(y_test, class_test_y, labels=classes)

# we get the confusion matrix function
# create confusion matrix for cm
# this would plot a confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()


# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

# select a train and test split data based on the participant
# we have 5 participants in the dataset
# we would train on 4 participants except participant A
# create a participant dataframe
# we drop the "category" and "set" columns from the dataframe
participant_df = df.drop(["category", "set"], axis=1)

# select our training data from all participants except participant "A"
X_train = participant_df[participant_df["participant"] != "A"].drop("label", axis=1)

# we are only taking the "label" column from the training data where participant is not "A"
y_train = participant_df[participant_df["participant"] != "A"]["label"]

# this is for "test" data
# this is our test data for participant "A"
X_test = participant_df[participant_df["participant"] == "A"].drop("label", axis=1)

# this is the label column for the test data where participant is "A"
y_test = participant_df[participant_df["participant"] == "A"]["label"]

# we remove the participant column from the training and test data
X_train = X_train.drop(["participant"], axis=1)
X_test = X_test.drop(["participant"], axis=1)

# we plot the train and test data to see the distribution of labels
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total", alpha=0.7
)
y_train.value_counts().plot(
    kind="bar", ax=ax, color="dodgerBlue", label="Train", alpha=0.7
)
y_test.value_counts().plot(
    kind="bar", ax=ax, color="royalBlue", label="Test", alpha=0.7
)
plt.legend()
plt.show()


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------


# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
