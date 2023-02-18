# Library imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from pathlib import Path
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


def get_confusion_matrix_loss(lm, sample_size, priors):
    loss = 0
    dlm = np.copy(lm)
    for i in range(np.shape(lm)[0]):
        for j in range(np.shape(lm)[1]):
            if i != j:
                loss += lm[i][j]
    for index in range(np.shape(lm)[0]):
        dlm[index] = [round(x / (priors[index] * sample_size), 4) for x in lm[index]]
    return dlm, loss / sample_size


def get_dataset_average(dataset):
    data_shape = np.shape(dataset)
    indexes = data_shape[1]
    data_avg = []
    for i in range(indexes):
        data_avg.append(np.average(dataset[:, i]))
    data_avg = (np.asarray(data_avg))
    return data_avg


def return_risk(shape, dataset, choice, lamda, M, C, P):
    risk = 0
    for element in range(shape):
        risk = risk + lamda[dataset, element] * P[element] * multivariate_normal.pdf(choice, M[element], C[element])
    return float(risk)


def classify_map(dataset, lamda, loss_matrix, M, C, P):
    return_dataframe = pd.DataFrame(dataset, columns=dataset.columns)
    return_dataframe['Guess'] = 0
    dataset_shape = np.shape(dataset)[0]
    sample_counter = 0
    for sample in dataset.values:
        print(f'Processing Sample: {sample_counter}')
        choice_list = []
        for prior in range(len(P)):
            choice_list.append(return_risk(len(P), prior, sample[:-2], lamda, M, C, P))
        choice = np.argmin(np.asarray(choice_list))

        if choice == sample[-2]:
            return_dataframe.at[sample_counter, 'Guess'] = 1
        # Compute loss matrix
        loss_matrix[sample[-2], choice] = loss_matrix[sample[-2], choice] + 1
        sample_counter += 1

    return return_dataframe, loss_matrix


def reduce_feature_size(dataset, n_features, class_prior):
    class_column = dataset[class_prior]
    dataset = dataset.drop(columns=[class_prior])
    PCA3 = PCA(n_components=n_features).fit_transform(dataset)
    column_names = []
    for i in range(n_features):
        column_names.append(f'Principle Component {i + 1}')
    PCA3_dataframe = pd.DataFrame(data=PCA3,
                                  columns=column_names)
    PCA3_dataframe[class_prior] = class_column
    return PCA3_dataframe


def generate_graphs(dataset, class_prior, confusion_mat):
    class_column = dataset[class_prior]
    guess_column = dataset['Guess']
    dataset = dataset.drop(columns=[class_prior])
    dataset = dataset.drop(columns=['Guess'])
    PCA3 = PCA(n_components=3).fit_transform(dataset)
    PCA3_dataframe = pd.DataFrame(data=PCA3,
                                  columns=['Principle Component 1', 'Principle Component 2', 'Principle Component 3'])
    PCA3_dataframe[class_prior] = class_column
    PCA3_dataframe['Guess'] = guess_column

    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'gold', 'cadetblue', 'thistle', 'seagreen', 'saddlebrown', 'orangered']
    point_list = ['.', 'v', '^', '3', 's', 'p', 'P', '*', 'x', 'D', 'h', 'H', '+', 'd']

    # Split the PCA
    print(f'Splitting PCA by Labels')
    correct_PCA = []
    incorrect_PCA = []
    split_PCA = []
    for label in range(min_value, max_value):
        interest = (PCA3_dataframe[PCA3_dataframe[class_prior] == label]).drop(columns=[class_prior])
        correct_PCA.append(PCA3_dataframe[(PCA3_dataframe[class_prior] == label) & (PCA3_dataframe['Guess'] == 1)])
        incorrect_PCA.append(PCA3_dataframe[(PCA3_dataframe[class_prior] == label) & (PCA3_dataframe['Guess'] == 0)])
        split_PCA.append(interest)
    print(f'Splitting PCA by Labels Done...')

    plt.figure(0)
    plt.title('PCA Analysis, 2-D')
    color_count = 0
    for class_set in split_PCA:
        plt.scatter(class_set['Principle Component 1'],
                    class_set['Principle Component 2'],
                    color=color_list[color_count],
                    label=f'Class {color_count + min_value}')
        color_count += 1
    plt.legend()
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(projection='3d')
    plt.title('PCA Analysis, 3-D')
    color_count = 0
    for class_set in split_PCA:
        ax.scatter(class_set['Principle Component 1'],
                   class_set['Principle Component 2'],
                   class_set['Principle Component 3'],
                   color=color_list[color_count],
                   label=f'Class {color_count + min_value}')
        color_count += 1
    ax.set_xlabel('Principle Component 1')
    ax.set_ylabel('Principle Component 2')
    ax.set_zlabel('Principle Component 3')
    ax.legend()

    plt.figure(2)
    plt.title('PCA Analysis Results, 2-D')
    color_count = 0
    for class_set in correct_PCA:
        plt.scatter(class_set['Principle Component 1'],
                    class_set['Principle Component 2'],
                    marker=point_list[color_count],
                    color='g',
                    label=f'Correct Class {color_count + min_value}')
        color_count += 1
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')

    color_count = 0
    for class_set in incorrect_PCA:
        plt.scatter(class_set['Principle Component 1'],
                    class_set['Principle Component 2'],
                    marker=point_list[color_count],
                    color='r',
                    label=f'Incorrect Class {color_count + min_value}')
        color_count += 1
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')
    plt.legend()

    fig2 = plt.figure(3)
    ax = fig2.add_subplot(projection='3d')
    plt.title('PCA Analysis Results, 3-D')
    color_count = 0
    for class_set in correct_PCA:
        ax.scatter(class_set['Principle Component 1'],
                   class_set['Principle Component 2'],
                   class_set['Principle Component 3'],
                   color='g',
                   marker=point_list[color_count + min_value],
                   label=f'Class {color_count + min_value}')
        color_count += 1
    ax.set_xlabel('Principle Component 1')
    ax.set_ylabel('Principle Component 2')
    ax.set_zlabel('Principle Component 3')

    color_count = 0
    for class_set in incorrect_PCA:
        ax.scatter(class_set['Principle Component 1'],
                   class_set['Principle Component 2'],
                   class_set['Principle Component 3'],
                   color='r',
                   marker=point_list[color_count + min_value],
                   label=f'Class {color_count + min_value}')
        color_count += 1
    ax.set_xlabel('Principle Component 1')
    ax.set_ylabel('Principle Component 2')
    ax.set_zlabel('Principle Component 3')
    ax.legend()

    fig3, ax = plt.subplots()
    ax.matshow(confusion_mat, cmap=plt.cm.Blues)
    for i in range(np.shape(confusion_mat)[0]):
        for j in range(np.shape(confusion_mat)[1]):
            c = confusion_mat[j, i]
            ax.text(i, j, str(c), va='center', ha='center')

    plt.show()
    return PCA3_dataframe


if __name__ == '__main__':
    shrink_feature_set = False
    wine_dataset_path = Path(
        r'C:\Users\David Hunter\OneDrive\Northeastern Classes\Graduate\EECE5644MachineLearning\Homework1\HumanActivityData.csv')
    wine_dataset = read_csv(wine_dataset_path, delimiter=',')
    entries = wine_dataset.shape[0]
    prior_variable = 'Activity'
    min_value = wine_dataset[prior_variable].min()
    max_value = wine_dataset[prior_variable].max() + 1
    print(f'Wine Dataset Loaded...\nTotal Entries: {entries}')

    print(f'Calculating Class Priors')
    wine_priors = []
    for score in range(min_value, max_value):
        wine_priors.append((wine_dataset[prior_variable] == score).sum() / entries)
    print(f'Class Priors Calculated (Assuming Scores Are Classes)...')

    if shrink_feature_set:
        wine_dataset = reduce_feature_size(wine_dataset, 10, prior_variable)

    features = wine_dataset.shape[1] - 1
    regularization_alpha = 0.5
    labeled_dataset = []

    averaged_dataset = []
    dataset_covariances = []
    regularized_dataset_covariances = []
    correct_list = []
    incorrect_list = []
    total_correct = 0
    total_incorrect = 0
    id_matrix = np.identity(features)
    loss_matrix = np.zeros((max_value - min_value, max_value - min_value))

    # Normalize the data
    class_column = wine_dataset[prior_variable]
    wine_dataset = wine_dataset.drop(columns=[prior_variable])
    wine_dataset = pd.DataFrame(StandardScaler().fit_transform(wine_dataset.loc[:].values),
                                columns=wine_dataset.columns)
    wine_dataset[prior_variable] = class_column

    # Split the dataset up by labels
    print(f'Splitting Dataset by Labels')
    for label in range(min_value, max_value):
        interest = (wine_dataset[wine_dataset[prior_variable] == label]).drop(columns=[prior_variable])
        labeled_dataset.append(interest)
    print(f'Splitting Dataset by Labels Done...')
    # Average the values in each dataset, save to new averaged list

    print(f'Averaging Label Values')
    for label in labeled_dataset:
        current_label_average = []
        for labelName in label:
            current_label_average.append(label[labelName].mean())
        averaged_dataset.append(current_label_average)
    print(f'Averaging of values done...')

    print(f'Calculating Covariance Matrices and Regularization Factor')
    for label in labeled_dataset:
        current_covariance_matrix = label.cov()
        dataset_covariances.append(current_covariance_matrix.values)
        lmda = regularization_alpha * np.trace(current_covariance_matrix.to_numpy()) / np.linalg.matrix_rank(
            current_covariance_matrix.to_numpy())
        current_covariances_regularized = current_covariance_matrix + lmda * id_matrix
        regularized_dataset_covariances.append(current_covariances_regularized)
    print(f'Calculating Covariance Matrices and Regularization Factor Done...')

    # Recombine the dataframes
    temp_dataframe = pd.DataFrame(columns=wine_dataset.columns)
    label_counter = 0
    for label in labeled_dataset:
        label[prior_variable] = label_counter
        temp_dataframe = temp_dataframe.append(label, ignore_index=True)
        label_counter += 1
    labeled_dataset = temp_dataframe

    print(f'Generating Lambda Matrix')
    # construct lambda matrix (Assuming no loss case)
    lambda_matrix = np.ones((len(wine_priors), len(wine_priors)))
    lambda_matrix[np.diag_indices_from(lambda_matrix)] = 0  # TODO: FIGURE OUT WHY np.fill_diagonal CREATES A NONE TYPE
    print(f'Generating Lambda Matrix Done...')

    print(f'Testing on MAP Classifier')
    # Test sample-map
    guess_dataset, loss_matrix_update = classify_map(labeled_dataset, lambda_matrix, loss_matrix, averaged_dataset,
                                                     regularized_dataset_covariances, wine_priors)
    # guess_dataset, loss_matrix_update = classify_map_mat(labeled_dataset, lambda_matrix, loss_matrix, averaged_dataset, regularized_dataset_covariances, wine_priors)
    print(f'Testing on MAP Classifier Done...')
    print(f'Sum of loss_matrix_update: {np.sum(loss_matrix_update)}')
    print(f'_________________________________________')
    print(f'Calculating Number Correct and Number Incorrect Done...')

    print(f'Calculating Confusion Matrix')
    display_confusion_matrix, loss = get_confusion_matrix_loss(loss_matrix_update, entries, wine_priors)
    rounded_loss = round(loss, 4)
    print(f'Loss: {rounded_loss * 100}%')
    print(f'Confusion Matrix:\n{display_confusion_matrix}')

    pca_data = generate_graphs(guess_dataset, prior_variable, display_confusion_matrix)
