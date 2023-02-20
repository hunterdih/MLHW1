# Library imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from pathlib import Path
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

REG_ALPHA = 0.025

def load_and_process_data(data_path='0', delim=',', normalize=True):
    if data_path == '0':
        print(f'ERROR: No Dataset Passed')
    data = read_csv(data_path, delimiter=delim)  # Load the data into a pandas dataframe
    class_column_name = data.columns.values.tolist()[
        -1]  # Get the name of the labels column (should be the last column)
    min_class_number = data[class_column_name].min()  # Get min class index for counting
    max_class_number = data[class_column_name].max() + 1  # Get max class index for counting
    print(f'{min_class_number=}, {max_class_number=}, entries={data.shape[0]}')

    print(f'Finding Class Priors...')
    #  Determine the class priors
    priors = []
    for prior in range(min_class_number, max_class_number):
        priors.append(
            (data[class_column_name] == prior).sum() / data.shape[0])  # Average occurrences of given class
    print(f'Class Priors Determined...')
    print(f'\n{priors=}')

    if normalize:
        print(f'Normalizing Data to Zero Mean, 1 Cov')
        class_column = data[class_column_name]
        data = data.drop(columns=[class_column_name])
        data = pd.DataFrame(StandardScaler().fit_transform(data.loc[:].values),
                            columns=data.columns)
        data[class_column_name] = class_column
        print(f'Data Normalized...')

    # Calculate the number of features for identity matrix
    features = data.shape[1] - 1  # Subtract 1 to exclude the class labels
    id_matrix = np.identity(features)
    class_means = []
    class_covs = []
    class_covs_reg = []

    print(f'Calculating Class Means and Covariances')
    for prior in range(min_class_number, max_class_number):
        interest_set = data.loc[data[class_column_name] == prior].drop(class_column_name, axis=1)
        class_means.append(interest_set.mean().values)
        class_covs.append(interest_set.cov())
        lmda = REG_ALPHA * np.trace(interest_set.cov().to_numpy()) / np.linalg.matrix_rank(
            interest_set.cov().to_numpy())
        class_covs_reg.append((interest_set.cov() + lmda * id_matrix))
    class_means = np.asarray(class_means)
    class_covs = np.asarray(class_covs)
    class_covs_reg = np.asarray(class_covs_reg)
    print(f'Means and Covariances Calculated...')

    return data, class_means, class_covs, class_covs_reg, priors


def map_classify(data, lmda_mat, class_means, class_covs, priors):
    loss_mat = np.zeros((len(priors), len(priors)))

    class_column_name = data.columns.values.tolist()[-1]
    min_class_number = data[class_column_name].min()  # Get min class index for counting
    max_class_number = data[class_column_name].max()
    class_column = data[class_column_name]
    return_dataframe = pd.DataFrame(data, columns=data.columns)
    return_dataframe = return_dataframe

    data_select = return_dataframe.values[:, :-1]
    pre_loss_values = []
    for prior in range(len(priors)):
        lmda_select = lmda_mat[prior]

        gau_values = np.asarray(multivariate_normal.pdf(data_select, class_means[prior], class_covs[prior]).tolist())
        hold_mat = np.zeros_like(gau_values)
        for risk_prior in range(len(priors)):
            mean_select = class_means[prior]
            covs_select = class_covs[prior]
            prior_select = priors[prior]
            gau_values = np.asarray(multivariate_normal.pdf(data_select, mean_select, covs_select).tolist())
            hold_mat = hold_mat + prior_select * lmda_select[risk_prior] * gau_values

        pre_loss_values.append(hold_mat.tolist())

    risk_dataframe = pd.DataFrame(pre_loss_values).T
    risk_dataframe[class_column_name] = class_column
    class_names = np.arange(min_class_number, max_class_number + 1, 1)
    class_indexes = np.arange(0, len(class_names), 1)
    choice_list = []
    guess_list = []
    for sample in risk_dataframe.values:
        choice = class_names[np.argmax(np.asarray(sample[:-1]))]

        if choice != sample[-1]:
            guess_list.append(0)
        else:
            guess_list.append(1)
        interest_index_true = np.where(class_names == sample[-1])
        interest_index_guess = np.where(class_names == choice)
        loss_mat[interest_index_true, interest_index_guess] = loss_mat[interest_index_true, interest_index_guess] + 1
        choice_list.append(choice)
    return_dataframe['choice'] = choice_list
    return_dataframe['guess'] = guess_list
    print(f'Risk Analyzed')

    return return_dataframe, loss_mat


def get_confusion_matrix_loss(lm, priors):
    sample_size = lm.sum()
    ls = 0
    dlm = np.copy(lm)
    for i in range(np.shape(lm)[0]):
        for j in range(np.shape(lm)[1]):
            if i != j:
                ls += lm[i][j]
    for index in range(np.shape(lm)[0]):
        dlm[index] = [round(x / (priors[index] * sample_size), 4) for x in lm[index]]
    return dlm, ls / sample_size


def generate_graphs(data, confusion_mat):
    class_column_name = data.columns.values.tolist()[-3]
    class_column = data[class_column_name]
    min_class_number = data[class_column_name].min()  # Get min class index for counting
    max_class_number = data[class_column_name].max()

    guess_column = data['guess']
    data = data.drop(columns=[class_column_name, 'guess', 'choice'])
    PCA3 = PCA(n_components=3).fit_transform(data)
    PCA3_dataframe = pd.DataFrame(data=PCA3,
                                  columns=['Principle Component 1', 'Principle Component 2', 'Principle Component 3'])
    PCA3_dataframe[class_column_name] = class_column
    PCA3_dataframe['guess'] = guess_column

    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'gold', 'cadetblue', 'thistle', 'seagreen', 'saddlebrown', 'orangered']
    point_list = ['.', 'v', '^', '3', 's', 'p', 'P', '*', 'x', 'D', 'h', 'H', '+', 'd']

    # Split the PCA
    print(f'Splitting PCA by Labels')
    correct_PCA = []
    incorrect_PCA = []
    split_PCA = []

    for label in range(min_class_number, max_class_number + 1):
        interest = (PCA3_dataframe[PCA3_dataframe[class_column_name] == label]).drop(columns=[class_column_name])
        correct_PCA.append(
            PCA3_dataframe[(PCA3_dataframe[class_column_name] == label) & (PCA3_dataframe['guess'] == 1)])
        incorrect_PCA.append(
            PCA3_dataframe[(PCA3_dataframe[class_column_name] == label) & (PCA3_dataframe['guess'] == 0)])
        split_PCA.append(interest)
    print(f'Splitting PCA by Labels Done...')

    plt.figure(0)
    plt.title('PCA Analysis, 2-D')
    color_count = 0
    for class_set in split_PCA:
        plt.scatter(class_set['Principle Component 1'],
                    class_set['Principle Component 2'],
                    color=color_list[color_count],
                    label=f'Class {color_count + min_class_number}')
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
                   label=f'Class {color_count + min_class_number}')
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
                    label=f'Correct Class {color_count + min_class_number}')
        color_count += 1
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')

    color_count = 0
    for class_set in incorrect_PCA:
        plt.scatter(class_set['Principle Component 1'],
                    class_set['Principle Component 2'],
                    marker=point_list[color_count],
                    color='r',
                    label=f'Incorrect Class {color_count + min_class_number}')
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
                   marker=point_list[color_count + min_class_number],
                   label=f'Class {color_count + min_class_number}')
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
                   marker=point_list[color_count + min_class_number],
                   label=f'Class {color_count + min_class_number}')
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
    # CONFIGURATIONS FOR DATASET SPECIFIC FUNCTIONS
    optimal_reg = False
    normalize_inputs = True
    # Import the dataset
    dataset_path = Path(
        r'C:\Users\David Hunter\OneDrive\Northeastern Classes\Graduate\EECE5644MachineLearning\Homework1\HumanActivityData.csv')
    reg_alpha_list = []
    if optimal_reg:
        for norm in range(1, 50):  # Path to the dataset
            REG_ALPHA = norm * 0.0001
            dataset, means, covariances, reg_covariances, class_priors = load_and_process_data(dataset_path, ',', normalize=normalize_inputs)

            # Generate the lambda matrix
            lambda_matrix = np.ones((len(class_priors), len(class_priors)))
            lambda_matrix[np.diag_indices_from(lambda_matrix)] = 0

            dataset, loss_matrix = map_classify(dataset, lambda_matrix, means, reg_covariances, class_priors)
            display_confusion_matrix, loss = get_confusion_matrix_loss(loss_matrix, class_priors)
            reg_alpha_list.append(loss)

        REG_ALPHA = reg_alpha_list.index(min(reg_alpha_list))*0.001
        print(f'{REG_ALPHA=}, {reg_alpha_list=}')
    dataset, means, covariances, reg_covariances, class_priors = load_and_process_data(dataset_path, ',', normalize=normalize_inputs)

    # Generate the lambda matrix
    lambda_matrix = np.ones((len(class_priors), len(class_priors)))
    lambda_matrix[np.diag_indices_from(lambda_matrix)] = 0

    dataset, loss_matrix = map_classify(dataset, lambda_matrix, means, reg_covariances, class_priors)
    display_confusion_matrix, loss = get_confusion_matrix_loss(loss_matrix, class_priors)
    rounded_loss = round(loss, 4)
    print(f'Loss: {rounded_loss * 100}%')
    print(f'Confusion Matrix:\n{display_confusion_matrix}')

    pca_data = generate_graphs(dataset, display_confusion_matrix)
