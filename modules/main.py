# ML Test Main File:

# Imports:
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Importing TSNE:
from sklearn.manifold import TSNE

# KNN, KFold and sklearn general:
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

# Evaluation Metrics:
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

# Utils:
import data_utils

# Define results output save_path:
SAVE_PATH = f'{Path(__file__).parent.parent}/results_1.csv'

#################################################################################################################################
# ### Retrieving Data and Exploration:

# Loading data from pickle file:
pickle_file_path = f'{Path(__file__).parent.parent}/mini_gm_public_v0.1.p'
data = data_utils.retrieve_data_from_pickle(pickle_file_path)

# -	The structure of the dictionary saved in the pickle file:
# {'syndrome_id': { 'subject_id': {'image_id': [320x1 encoding]}}}

# Let's organize data in a set with imgs and correspondent Syndrome Ids:
organized_data = []

for syndrome_id, value in data.items():
    for subject_id, img_id in value.items():
        for id, img in img_id.items():
            organized_data.append([syndrome_id, subject_id, id, np.array(img)])

# Convert it to pandas dataframe and visualize:
data_df = pd.DataFrame(organized_data)
data_df.columns = ['syndrome_id', 'subject_id', 'img_id','img']

# Let's Drop the irrelevant data and keep our inputs (imgs) and outputs (syndrome ids):
data_df.drop(columns=['subject_id', 'subject_id', 'img_id'], inplace=True)
#################################################################################################################################

# a) Plotting t-SNE of the inputs, explaining the statistics and the data:

# t-SNE, which stands for t-Distributed Stochastic Neighbor Embedding, is a technique in machine learning and data visualization 
# for reducing the dimensionality of high-dimensional data while preserving its structure and relationships.

"""
    Apply t-SNE for dimensionality reduction:

    Let's use n_components as 2 as a good default as we are dealing with an img, and we are able to visualize.
     
    Perplexity is a hyperparameter in t-SNE that balances the local and global aspects of the data.
    It determines the effective number of neighbors that each point is compared to during optimization.
    Perplexity should typically be set between 5 and 50.
     
    A lower perplexity value places more emphasis on local relationships, while a higher perplexity value considers more global 
    relationships. It's recommended to try a range of perplexity values and evaluate the resulting visualizations to determine the 
    most suitable value. Common choices for perplexity include values like 5, 10, 20, 30, and 50, **let's check 10, 30 and 50 as 
    it's values. random_state=42 as Douglas Adams would advise so.

"""

print('Plotting TSNE curves for input data...')

# Defining t-sne for n_components=2:
tsne_10 = TSNE(n_components=2, perplexity=10, random_state=42)
tsne_30 = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_50 = TSNE(n_components=2, perplexity=50, random_state=42)

# Let's apply it to our data:
data_tsne_10 = tsne_10.fit_transform(np.array(data_df['img'].tolist()))
data_tsne_30 = tsne_30.fit_transform(np.array(data_df['img'].tolist()))
data_tsne_50 = tsne_50.fit_transform(np.array(data_df['img'].tolist()))

# Let's get some colors for each unique syndrome:
colors = {syndrome_id:[random.randint(0,1), random.randint(0,1), random.randint(0,1)] for syndrome_id in data_df['syndrome_id'].unique()}

# Get colors for each instance of data:
data_colors = [colors[syndrome_id] for syndrome_id in data_df['syndrome_id']]
labels = ['Perplexity - 10', 'Perplexity - 30', 'Perplexity - 50']

# Plotting TSNE data:
for i, data in enumerate([data_tsne_10, data_tsne_30, data_tsne_50]):
    data_utils.plot_TSNE(data, data_colors, labels[i])
    
#################################################################################################################################

# b) Do a 10 fold cross validation for the following steps:
# -	Calculate cosine distance from each test set vector to the gallery vectors
# -	Calculate euclidean distance from each test set vector to the gallery vectors
# -	Classify each image (vector) or each subject to syndrome Ids based on KNN algorithm for both cosine and euclidean distances. 
    
print('Encoding data...')

# Etract Input X and output y data:
X_data = np.array(data_df['img'].tolist(), dtype=np.int32)
y_data = data_df['syndrome_id']

# Let's encode our syndrome_id strings data:
encoder = LabelEncoder()
y_data = encoder.fit_transform(y_data)
y_data = y_data.reshape(-1,1)

# Visualize encoding:
label_encoded_dict = {label: encoded for label, encoded in zip(encoder.classes_, encoder.transform(encoder.classes_))}
print('Encoded Classes:', label_encoded_dict)

#################################################################################################################################
# KNN Training:

# OneVsRestClassifier will avoid some issues with multiclass labeling in scikit-learn
# This approach is often used when the native classifier supports only binary classification
# and it's needed to apply to multi-class classification. 

# Instances of KNN, Let's use n_neighbors=15 after some testing.
knn_cosine = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=15))
knn_euclidean = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=15))

# Initialize variables to store values for each distance metric:
cosine_accuracies = []
cosine_f1_scores = []
euclidean_accuracies = []
euclidean_f1_scores = []
cosine_roc_aucs = []
euclidean_roc_aucs = []

# Define the number of folds as 10 for cross-validation.
# Let's use stratified to keep class balance:
kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold = 0

print('Running 10-Fold Cross Validation for both Cosine Distances and Euclidean Distances KNN.')

# Performing 10-fold cross-validation:
for train_index, test_index in kfolds.split(X_data, y_data):

    # Get fold data by index:
    X_train_fold, X_test_fold = X_data[train_index], X_data[test_index]
    y_train_fold, y_test_fold = y_data[train_index], y_data[test_index]

    # Calculating cosine distance between train and test:
    cosine_dists_train = cosine_distances(X_train_fold) # To train
    cosine_dists_test = cosine_distances(X_test_fold, X_train_fold) # To test
    
    # Calculating euclidean distance between train and test:
    euclidean_dists_train = euclidean_distances(X_train_fold) # To train
    euclidean_dists_test = euclidean_distances(X_test_fold, X_train_fold) # To test

    # Fit KNN models with calculated distances:
    knn_cosine.fit(cosine_dists_train, y_train_fold)
    knn_euclidean.fit(euclidean_dists_train, y_train_fold)

    # Predict for test using cosine distance:
    y_pred_cosine = knn_cosine.predict(cosine_dists_test)

    # Metrics for cosine distances:
    cosine_accuracy = accuracy_score(y_test_fold, y_pred_cosine)
    cosine_accuracies.append(cosine_accuracy)
    cosine_f1 = f1_score(y_test_fold, y_pred_cosine, average='macro')
    cosine_f1_scores.append(cosine_f1)

    # Predict using euclidean distance:
    y_pred_euclidean = knn_euclidean.predict(euclidean_dists_test)

    # Metrics for euclidean distances:
    euclidean_accuracy = accuracy_score(y_test_fold, y_pred_euclidean)
    euclidean_accuracies.append(euclidean_accuracy)
    euclidean_f1 = f1_score(y_test_fold, y_pred_cosine, average='macro')
    euclidean_f1_scores.append(euclidean_f1)

    # Print metrics for current fold:
    print('==='*30)
    print(f'Fold: {fold+1}')
    print(f'Cosine Distance - KNN - Accuracy: {cosine_accuracy}')
    print(f'Euclidean Distance - KNN - Accuracy: {euclidean_accuracy}')
    print(f'Cosine Distance - KNN - F1 Score: {cosine_f1}')
    print(f'Euclidean Distance - KNN - F1 Score: {euclidean_f1}')

    # Increment Fold:
    fold += 1

    # Roc Auc Curve for Cosine Distances:
    # Get classes probabilities:
    y_score_cosine = knn_cosine.predict_proba(cosine_dists_test)

    fpr_cosine = {} # false positives rate
    tpr_cosine = {} # true positives rate
    roc_auc_cosine = {}

    # For each class in our data:
    for i in range(len(knn_cosine.classes_)):
        fpr_cosine[i], tpr_cosine[i], _ = roc_curve(y_test_fold == knn_cosine.classes_[i], y_score_cosine[:, i])
        roc_auc_cosine[i] = auc(fpr_cosine[i], tpr_cosine[i])

    # Retrieve data for this fold:
    cosine_roc_aucs.append(roc_auc_cosine)

    # Roc Auc Curve for Euclidean Distances:
    # Get classes probabilities:
    y_score_euclidean = knn_cosine.predict_proba(euclidean_dists_test)

    fpr_euclidean = {} # false positives rate
    tpr_euclidean = {} # true positives rate
    roc_auc_euclidean = {}

    # For each class in our data:
    for i in range(len(knn_euclidean.classes_)):
        fpr_euclidean[i], tpr_euclidean[i], _ = roc_curve(y_test_fold == knn_euclidean.classes_[i], y_score_euclidean[:, i])
        roc_auc_euclidean[i] = auc(fpr_euclidean[i], tpr_euclidean[i])

    # Retrieve data for this fold:
    euclidean_roc_aucs.append(roc_auc_euclidean)

#################################################################################################################################
# Evaluation Metrics:

# Calculate average accuracies:
avg_cosine_accuracy = np.mean(cosine_accuracies)
avg_euclidean_accuracy = np.mean(euclidean_accuracies)

# Calculate average F1s:
avg_cosine_f1_scores = np.mean(cosine_f1_scores)
avg_euclidean_f1_scores = np.mean(euclidean_f1_scores)

print('==='*40)
print("Average Cosine Accuracy:", round(avg_cosine_accuracy, 2))
print("Average Euclidean Accuracy:", round(avg_euclidean_accuracy, 2), '\n')
print('==='*40)
print("Average Cosine F1 score:", round(avg_cosine_f1_scores, 2))
print("Average Euclidean F1 score:", round(avg_euclidean_f1_scores, 2))
print('==='*40)

#################################################################################################################################

# Saving Data:
# c) Create automatic tables in a txt / pdf file for both algorithms, to enable comparison (please specify top-k, AUC etc.)
print('Organizing results data to save...')

# Organizing data from gathered metrics:
output_data = [[x+1, cosine_accuracies[x], euclidean_accuracies[x], cosine_f1_scores[x], euclidean_f1_scores[x],
                cosine_roc_aucs[x], euclidean_roc_aucs[x]] for x in range(10)]

# Transform data into dataframe:
output_data_df = pd.DataFrame(output_data)
output_data_df.columns = ['Fold', 'Cosine Dist - Accuracy', 'Euclidean Dist - Accuracy',
                          'Cosine Dist - F1 Score', 'Euclidean Dist - F1 Score', 
                          'Cosine Dist - ROC AUC Scores', 'Euclidean Dist - ROC AUC Scores', ]

# Let's sort our data:
output_data_df = output_data_df.sort_values(by=['Cosine Dist - Accuracy', 'Euclidean Dist - Accuracy'], ascending=[False, False])

# Round values in specific columns using lambda function
output_data_df[['Cosine Dist - Accuracy', 'Euclidean Dist - Accuracy']] = output_data_df[['Cosine Dist - Accuracy', 'Euclidean Dist - Accuracy']].apply(lambda x: round(x, 2))
output_data_df[['Cosine Dist - F1 Score', 'Euclidean Dist - F1 Score']] = output_data_df[['Cosine Dist - F1 Score', 'Euclidean Dist - F1 Score']].apply(lambda x: round(x, 2))
output_data_df.head(10)

# Now let's save our data to csv file, as results.csv:
output_data_df.to_csv(SAVE_PATH, sep=',', index=False)
print(f'... Results saved to {SAVE_PATH}')
#################################################################################################################################

# Roc Auc Curve:
# d) Create an ROC AUC graph comparing both algorithms (2 outputs in the same graph, averaged across gallery / test splits

print('Plotting ROC AUC Curves...')

# Let's plot the Roc Auc Curve for the first fold (0):
fig, axes = plt.subplots(1, 1, figsize=(16, 4))
    
axes.plot(knn_cosine.classes_, cosine_roc_aucs[0].values(), color='b', label='Cosine Distance KNN')
axes.plot(knn_euclidean.classes_, euclidean_roc_aucs[0].values(), color='r', label='Euclidean Distance KNN')
axes.set_xlabel('Classes')
axes.set_ylabel('Score')
axes.set_title(f'ROC Curve - First Fold')
axes.legend(loc='upper right')
axes.set_xticks(range(len(knn_cosine.classes_)))
axes.set_xticklabels(label_encoded_dict.keys())
    
plt.show()

# Let's plot the Roc Auc Curve for the third fold (2):
fig, axes = plt.subplots(1, 1, figsize=(16, 4))
    
axes.plot(knn_cosine.classes_, cosine_roc_aucs[2].values(), color='b', label='Cosine Distance KNN')
axes.plot(knn_euclidean.classes_, euclidean_roc_aucs[2].values(), color='r', label='Euclidean Distance KNN')
axes.set_xlabel('Classes')
axes.set_ylabel('Score')
axes.set_title(f'ROC Curve - Third Fold')
axes.legend(loc='upper right')
axes.set_xticks(range(len(knn_cosine.classes_)))
axes.set_xticklabels(label_encoded_dict.keys())
    
plt.show()

# Let's plot the Roc Auc Curve for the Seventh fold (6):
fig, axes = plt.subplots(1, 1, figsize=(16, 4))
    
axes.plot(knn_cosine.classes_, cosine_roc_aucs[6].values(), color='b', label='Cosine Distance KNN')
axes.plot(knn_euclidean.classes_, euclidean_roc_aucs[6].values(), color='r', label='Euclidean Distance KNN')
axes.set_xlabel('Classes')
axes.set_ylabel('Score')
axes.set_title(f'ROC Curve - Seventh Fold')
axes.legend(loc='upper right')
axes.set_xticks(range(len(knn_cosine.classes_)))
axes.set_xticklabels(label_encoded_dict.keys())
    
plt.show()

#################################################################################################################################






