# Library imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

samples = 10000
PL1 = 0.3
PL2 = 0.3
PL3 = 0.4
m1 = np.array(([1, 1, 6]))
c1 = np.array(([[2, 0, 0],
                [0, 2, 0],
                [0, 0, 2]]))

m2 = np.array(([1, 1, 3]))
c2 = np.array(([1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]))

m3 = np.array(([3, 1, 1]))
c3 = np.array(([3, 0, 0],
               [0, 3, 0],
               [0, 0, 3]))

m4 = np.array(([3, 1, 1]))
c4 = np.array(([1.5, 0, 0],
               [0, 1.5, 0],
               [0, 0, 1.5]))

gauss_weights = 1


def return_risk(shape, dataset, choice, lamda, M, C, P):
    risk = 0
    for element in range(shape):
        risk = risk + lamda[dataset, element] * P[element] * multivariate_normal.pdf(choice, M[element], C[element])
    return float(risk)


def classify_map(dataset, gaussian, lamda, loss_matrix, M, C, P):
    correct = []
    incorrect = []
    dataset_shape = np.shape(dataset)[0]
    for point in dataset[gaussian]:
        choice1 = return_risk(dataset_shape, 0, point, lamda, M, C, P)
        choice2 = return_risk(dataset_shape, 1, point, lamda, M, C, P)
        choice3 = return_risk(dataset_shape, 2, point, lamda, M, C, P)

        choice = np.argmin([choice1, choice2, choice3])

        if choice == gaussian:
            correct.append(point)
        else:
            incorrect.append(point)
        # Compute loss matrix
        loss_matrix[gaussian, choice] = loss_matrix[gaussian, choice] + 1

    correct_x = []
    correct_y = []
    correct_z = []
    incorrect_x = []
    incorrect_y = []
    incorrect_z = []
    for point in correct:
        correct_x.append(point[0])
        correct_y.append(point[1])
        correct_z.append(point[2])
    for point in incorrect:
        incorrect_x.append(point[0])
        incorrect_y.append(point[1])
        incorrect_z.append(point[2])
    return np.asarray([correct_x, correct_y, correct_z]), np.asarray(
        [incorrect_x, incorrect_y, incorrect_z]), loss_matrix


def confusion_matrix(lm, sample_size=samples):
    loss = 0
    for i in range(np.shape(lm)[0]):
        for j in range(np.shape(lm)[1]):
            if i != j:
                loss += lm[i][j]
    print(f'Total Loss: {loss / sample_size}')
    for index in range(np.shape(lm)[0]):
        lm[index] = [x / (p[index] * samples) for x in lm[index]]
    return lm


if __name__ == '__main__':
    print(f'Starting Homework1 Question 2')
    dataset_list = []
    # Generate the dataset (10000 total samples)
    m = [m1, m2, m3 + m4]
    c = [c1, c2, c3 + c4]
    p = [PL1, PL2, PL3]

    dataset_list.append(np.asarray(np.random.multivariate_normal(m[0], c[0], int(p[0] * samples))))
    dataset_list.append(np.asarray(np.random.multivariate_normal(m[1], c[1], int(p[1] * samples))))
    dataset_list.append(np.asarray(np.random.multivariate_normal(m[2], c[2], int(p[2] * samples))))

    # Specify the decision rule that achieves minimum probability of error (identity matrix with 1 = 0 and 0 = 1)
    lamda_matrix_0 = np.array(([0, 1, 1],
                               [1, 0, 1],
                               [1, 1, 0]))

    fig0 = plt.figure(0)
    ax = fig0.add_subplot(projection='3d')
    plt.title('Data Distribution of 3 Gaussian Datasets')
    ax.scatter(dataset_list[0][:, 0], dataset_list[0][:, 1], dataset_list[0][:, 2], marker='o', label="Class0")
    ax.scatter(dataset_list[1][:, 0], dataset_list[1][:, 1], dataset_list[1][:, 2], marker='x', label="Class1'")
    ax.scatter(dataset_list[2][:, 0], dataset_list[2][:, 1], dataset_list[2][:, 2], marker='^', label="class2")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()

    # 0-1 Loss specifies a special case of the Bayes decision rule: MAP classifier
    # Need two functions, one to get the minimum risk, and one to get the MAP
    # Minimum risk = minRisk[x] = sum(lambda(D|x)*ClassPrior*p(x|mu, C))

    fig1 = plt.figure(1)
    loss_matrix_0 = np.zeros((3, 3))
    ax = fig1.add_subplot(projection='3d')
    plt.title('Correct Class Guesses of 3 Gaussian Datasets, No Loss')
    correct0, incorrect0, loss_matrix_0 = classify_map(dataset_list, 0, lamda_matrix_0, loss_matrix_0, m, c, p)
    correct1, incorrect1, loss_matrix_0 = classify_map(dataset_list, 1, lamda_matrix_0, loss_matrix_0, m, c, p)
    correct2, incorrect2, loss_matrix_0 = classify_map(dataset_list, 2, lamda_matrix_0, loss_matrix_0, m, c, p)
    ax.scatter(correct0[0], correct0[1], correct0[2], c='g', marker='o', label='Class 0 Correct Guesses')
    ax.scatter(incorrect0[0], incorrect0[1], incorrect0[2], c='r', marker='o', label='Class 0 incorrect Guesses')

    ax.scatter(correct1[0], correct1[1], correct1[2], c='g', marker='x', label='Class 1 Correct Guesses')
    ax.scatter(incorrect1[0], incorrect1[1], incorrect1[2], c='r', marker='x', label='Class 1 incorrect Guesses')

    ax.scatter(correct2[0], correct2[1], correct2[2], c='g', marker='^', label='Class 2 Correct Guesses')
    ax.scatter(incorrect2[0], incorrect2[1], incorrect2[2], c='r', marker='^', label='Class 2 incorrect Guesses')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Calculation of confusion matrix, inspired by summer 2020 problems, and provided matlab code
    print(f'PART A NO LOSS...')
    confusion = confusion_matrix(loss_matrix_0, samples)
    print(f'Confusion Matrix Estimation:\n{confusion}')

    # new loss matrices
    lamda_matrix_10 = np.array(([0, 1, 10],
                                [1, 0, 10],
                                [1, 1, 0]))

    lamda_matrix_100 = np.array(([0, 1, 100],
                                 [1, 0, 100],
                                 [1, 1, 0]))

    fig1 = plt.figure(2)
    loss_matrix_10 = np.zeros((3, 3))
    ax = fig1.add_subplot(projection='3d')
    plt.title('Correct Class Guesses of 3 Gaussian Datasets, 10 Times Loss')
    correct0, incorrect0, loss_matrix_10 = classify_map(dataset_list, 0, lamda_matrix_10, loss_matrix_10, m, c, p)
    correct1, incorrect1, loss_matrix_10 = classify_map(dataset_list, 1, lamda_matrix_10, loss_matrix_10, m, c, p)
    correct2, incorrect2, loss_matrix_10 = classify_map(dataset_list, 2, lamda_matrix_10, loss_matrix_10, m, c, p)
    ax.scatter(correct0[0], correct0[1], correct0[2], c='g', marker='o', label='Class 0 Correct Guesses')
    ax.scatter(incorrect0[0], incorrect0[1], incorrect0[2], c='r', marker='o', label='Class 0 incorrect Guesses')

    ax.scatter(correct1[0], correct1[1], correct1[2], c='g', marker='x', label='Class 1 Correct Guesses')
    ax.scatter(incorrect1[0], incorrect1[1], incorrect1[2], c='r', marker='x', label='Class 1 incorrect Guesses')

    ax.scatter(correct2[0], correct2[1], correct2[2], c='g', marker='^', label='Class 2 Correct Guesses')
    ax.scatter(incorrect2[0], incorrect2[1], incorrect2[2], c='r', marker='^', label='Class 2 incorrect Guesses')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    print(f'PART B 10x LOSS...')
    confusion = confusion_matrix(loss_matrix_10, samples)
    print(f'Confusion Matrix Estimation:\n{confusion}')

    fig1 = plt.figure(3)
    loss_matrix_100 = np.zeros((3, 3))
    ax = fig1.add_subplot(projection='3d')
    plt.title('Correct Class Guesses of 3 Gaussian Datasets, 100 Times Loss')
    correct0, incorrect0, loss_matrix_100 = classify_map(dataset_list, 0, lamda_matrix_100, loss_matrix_100, m, c, p)
    correct1, incorrect1, loss_matrix_100 = classify_map(dataset_list, 1, lamda_matrix_100, loss_matrix_100, m, c, p)
    correct2, incorrect2, loss_matrix_100 = classify_map(dataset_list, 2, lamda_matrix_100, loss_matrix_100, m, c, p)
    ax.scatter(correct0[0], correct0[1], correct0[2], c='g', marker='o', label='Class 0 Correct Guesses')
    ax.scatter(incorrect0[0], incorrect0[1], incorrect0[2], c='r', marker='o', label='Class 0 incorrect Guesses')

    ax.scatter(correct1[0], correct1[1], correct1[2], c='g', marker='x', label='Class 1 Correct Guesses')
    ax.scatter(incorrect1[0], incorrect1[1], incorrect1[2], c='r', marker='x', label='Class 1 incorrect Guesses')

    ax.scatter(correct2[0], correct2[1], correct2[2], c='g', marker='^', label='Class 2 Correct Guesses')
    ax.scatter(incorrect2[0], incorrect2[1], incorrect2[2], c='r', marker='^', label='Class 2 incorrect Guesses')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    print(f'PART B 100x LOSS...')
    confusion = confusion_matrix(loss_matrix_100, samples)
    print(f'Confusion Matrix Estimation:\n{confusion}')

    plt.show()
