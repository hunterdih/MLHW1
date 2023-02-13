# Library imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Question 1:
#
# The probability density function (pdf) for a 4-dimensional real-valued random vector X is as
# follows: p(x) = p(x|L = 0)P(L = 0) + p(x|L = 1)P(L = 1). Here L is the true class label that
# indicates which class-label-conditioned pdf generates the data.
# The class priors are P(L = 0) = 0.35 and P(L = 1) = 0.65. The class class-conditional pdfs are
# p(x|L = 0) = g(x|m0,C0) and p(x|L = 1) = g(x|m1,C1), where g(x|m,C) is a multivariate Gaus-
# sian probability density function with mean vector m and covariance matrix C. The parameters of
# the class-conditional Gaussian pdfs are:

# Important Constants:
m_0 = np.transpose(np.array([-0.5, -0.5, -0.5, -0.5]))

C_0 = 0.25 * np.array([[2, -0.5, 0.3, 0],
                       [-0.5, 1, -0.5, 0],
                       [0.3, -0.5, 1, 0],
                       [0, 0, 0, 2]])
m_1 = np.transpose(np.array([1, 1, 1, 1]))

C_1 = np.array([[1, 0.3, -0.2, 0],
                [0.3, 2, 0.3, 0],
                [-0.2, 0.3, 1, 0],
                [0, 0, 0, 3]])

samples = 10000
PL0 = 0.35
PL1 = 0.65

theoretical_gamma = PL0 / PL1


def get_discriminant(data0, data1, m0=m_0, C0=C_0, pl0=PL0, m1=m_1, C1=C_1, pl1=PL1, smpl=samples):
    disc_0 = np.array(
        multivariate_normal.pdf(data0[0], m1, C1) / multivariate_normal.pdf(data0[0], m0, C0))
    disc_1 = np.array(
        multivariate_normal.pdf(data1[0], m1, C1) / multivariate_normal.pdf(data1[0], m0, C0))

    for i in range(int(pl0 * smpl) - 1):
        disc_0 = np.append(disc_0,
                           multivariate_normal.pdf(data0[i], m1, C1) / multivariate_normal.pdf(data0[i], m0, C0))
    for i in range(int(pl1 * smpl) - 1):
        disc_1 = np.append(disc_1,
                           multivariate_normal.pdf(data1[i], m1, C1) / multivariate_normal.pdf(data1[i], m_0, C0))

    return disc_0, disc_1


def erm_classifier(disc_0, disc_1, pl0=PL0, pl1=PL1, smpls=samples):
    # sum of all discriminants
    disc_0 = list(disc_0)
    disc_1 = list(disc_1)
    discs = disc_0+disc_1
    # True positive probability
    tpp = []
    # False positive probability
    fpp = []
    # Error probability
    ep = []
    # g = gamma
    gammas = []
    for g in np.sort(discs):
        # P(D = 1|L = 1)
        tpc = ((disc_1 > g).sum() / (pl1 * smpls))
        # P(D = 1|L = 0)
        fpc = ((disc_0 > g).sum() / (pl0 * smpls))

        tpp.append(tpc)
        fpp.append(fpc)

        # (error; γ) = P(D = 1|L = 0; γ)P(L = 0) + P(D = 0|L = 1; γ)P(L = 1)
        ep.append((fpc * pl0) + ((1 - tpc) * pl1))
        gammas.append(g)

    return tpp, fpp, ep, gammas


def get_dataset_average(dataset):
    data_shape = np.shape(dataset)
    indexes = data_shape[1]
    data_avg = []
    for i in range(indexes):
        data_avg.append(np.average(dataset[:, i]))
    data_avg = (np.asarray(data_avg))
    return data_avg


if __name__ == '__main__':
    print(f'Parameters for Problem 1')
    print(f'm_0 = \n{m_0} \nof size {np.shape(m_0)}\n')
    print(f'm_0 = \n{C_0} \nof size {np.shape(C_0)}\n')
    print(f'm_0 = \n{m_1} \nof size {np.shape(m_1)}\n')
    print(f'm_0 = \n{C_1} \nof size {np.shape(C_1)}\n')

    # Generate the Dataset
    dataset0 = np.random.multivariate_normal(m_0, C_0, int(PL0 * samples))
    dataset1 = np.random.multivariate_normal(m_1, C_1, int(PL1 * samples))

    # Discriminant Score Calculations
    # Equates to the equation: p(x|L = 1)/p(x|L = 0) -> g(x|m0, c0)/g(x|m1, c1) = discriminant value
    # Can be though of as "what does my model think the choice should belong to".
    discriminant_0, discriminant_1 = get_discriminant(dataset0, dataset1, m_0, C_0, PL0, m_1, C_1, PL1, samples)

    true_positive_prob, false_positive_prob, error_prob, gamma = erm_classifier(discriminant_0, discriminant_1, PL0,
                                                                                PL1, samples)
    print(f'_________________________________________')
    print(f'Part A Outputs and Checks')
    print(f"True Positive Max = {max(true_positive_prob)}")
    print(f"True Positive Min = {min(true_positive_prob)}")
    print(f"False Positive Max = {max(false_positive_prob)}")
    print(f"False Positive Min = {min(false_positive_prob)}")

    min_error = min(error_prob)
    error_prob = np.asarray(error_prob)
    gamma_location = np.where(error_prob == min_error)[0][0]

    print(f'Gamma Value Selected: {gamma[gamma_location]}')
    print(f'Empirical Gamma: {theoretical_gamma}')
    print(f"Minimum Error = {min_error}")
    min_error_round = round(min_error, 3)
    print(f"Minimum Error Index = {gamma_location}")

    # PLOTTING THE ROC CURVE
    plt.figure(1)
    plt.title('ROC Curve for ERM Classification')
    plt.plot(false_positive_prob, true_positive_prob, label="ROC CURVE")
    plt.plot(false_positive_prob[gamma_location], true_positive_prob[gamma_location], 'go',
             label=f"Experimental Minimum: {min_error_round}")
    # Calculate for theoretical minimum error
    theo_true_positive = (discriminant_1 >= theoretical_gamma).sum() / (PL1 * samples)
    theo_false_positive = (discriminant_0 >= theoretical_gamma).sum() / (PL0 * samples)
    theo_min_error = (theo_false_positive * PL0) + ((1 - theo_true_positive) * PL1)
    theo_min_error_round = round(theo_min_error, 3)
    plt.plot(theo_false_positive, theo_true_positive, 'D', label=f'Theoretical Minimum: {theo_min_error_round}')
    plt.xlabel('False Detection')
    plt.ylabel('Correct Detection')
    plt.legend()

    # Part B
    # Class conditional pdf means are carried over from sample 1, datasets are also carried over from sample 1.
    # Create the diagonal covariance matrices
    C_b = np.array(([1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]))

    # Re-run the classifier by:
    # 1. re-calculating the discriminants
    # 2. finding the min-error
    # 3. visualizing the results

    # Discriminant Score Calculations
    # Equates to the equation: p(x|L = 1)/p(x|L = 0) -> g(x|m0, c0)/g(x|m1, c1) = discriminant value
    # Can be though of as "what does my model think the choice should belong to".
    discriminant_0, discriminant_1 = get_discriminant(dataset0, dataset1, C0=C_b, C1=C_b, smpl=samples)
    true_positive_prob, false_positive_prob, error_prob, gamma = erm_classifier(discriminant_0, discriminant_1, pl0=PL0,
                                                                                pl1=PL1, smpls=samples)

    print(f'_________________________________________')
    print(f'Part B Outputs and Checks')
    print(f"True Positive Max = {max(true_positive_prob)}")
    print(f"True Positive Min = {min(true_positive_prob)}")
    print(f"False Positive Max = {max(false_positive_prob)}")
    print(f"False Positive Min = {min(false_positive_prob)}")

    min_error = min(error_prob)
    error_prob = np.asarray(error_prob)
    gamma_location = np.where(error_prob == min_error)[0][0]

    print(f'Gamma Value Selected: {gamma[gamma_location]}')
    print(f'Empirical Gamma: {theoretical_gamma}')
    print(f"Minimum Error = {min_error}")
    min_error_round = round(min_error, 3)
    print(f"Minimum Error Index = {gamma_location}")

    # PLOTTING THE ROC CURVE
    plt.figure(2)
    plt.title('ROC Curve for ERM Classification, Incorrect Data Distribution Knowledge')
    plt.plot(false_positive_prob, true_positive_prob, label="ROC CURVE")
    plt.plot(false_positive_prob[gamma_location], true_positive_prob[gamma_location], 'go',
             label=f"Experimental Minimum: {min_error_round}")
    # Calculate for theoretical minimum error
    theo_true_positive = (discriminant_1 >= theoretical_gamma).sum() / (PL1 * samples)
    theo_false_positive = (discriminant_0 >= theoretical_gamma).sum() / (PL0 * samples)
    theo_min_error = (theo_false_positive * PL0) + ((1 - theo_true_positive) * PL1)
    theo_min_error_round = round(theo_min_error, 3)
    plt.plot(theo_false_positive, theo_true_positive, 'D', label=f'Theoretical Minimum: {theo_min_error_round}')
    plt.xlabel('False Detection')
    plt.ylabel('Correct Detection')
    plt.legend()
    #plt.show()


    # Part C Implementing a Fischer LDA based classifier
    # Creating a sample average estimator
    dataset0_avg = get_dataset_average(dataset0)
    dataset1_avg = get_dataset_average(dataset1)

    # Get covariance
    dataset0_cov = np.cov(dataset0, rowvar = False)
    dataset1_cov = np.cov(dataset1, rowvar = False)

    within_class_scatter = dataset0_cov + dataset1_cov
    between_class_scatter = (dataset0_avg - dataset1_avg) * (dataset0_avg - dataset1_avg).T

    eig_val, v = np.linalg.eig(np.linalg.inv(within_class_scatter)*between_class_scatter)

    # find max eigan value and its index for most important/relevent value/location
    max_eig_val_index = np.where(eig_val == np.max(eig_val))[0][0]
    wLDA = v[:, max_eig_val_index]
    # get determinants using max eigan value
    LDA0 = np.matmul(wLDA, dataset0.T)
    LDA1 = np.matmul(wLDA, dataset1.T)

    true_positive_prob, false_positive_prob, error_prob, gamma = erm_classifier(LDA0, LDA1, pl0=PL0,
                                                                                pl1=PL1, smpls=samples)

    print(f'_________________________________________')
    print(f'Part C Outputs and Checks')
    print(f"True Positive Max = {max(true_positive_prob)}")
    print(f"True Positive Min = {min(true_positive_prob)}")
    print(f"False Positive Max = {max(false_positive_prob)}")
    print(f"False Positive Min = {min(false_positive_prob)}")

    min_error = min(error_prob)
    error_prob = np.asarray(error_prob)
    gamma_location = np.where(error_prob == min_error)[0][0]

    print(f'Gamma Value Selected: {gamma[gamma_location]}')
    print(f'Empirical Gamma: {theoretical_gamma}')
    print(f"Minimum Error = {min_error}")
    min_error_round = round(min_error, 3)
    print(f"Minimum Error Index = {gamma_location}")

    # PLOTTING THE ROC CURVE
    plt.figure(3)
    plt.title('ROC Curve for Fischer LDA')
    plt.plot(false_positive_prob, true_positive_prob, label="ROC CURVE")
    plt.plot(false_positive_prob[gamma_location], true_positive_prob[gamma_location], 'go',
             label=f"Experimental Minimum: {min_error_round}")
    # Calculate for theoretical minimum error
    theo_true_positive = (LDA1 >= theoretical_gamma).sum() / (PL1 * samples)
    theo_false_positive = (LDA0 >= theoretical_gamma).sum() / (PL0 * samples)
    theo_min_error = (theo_false_positive * PL0) + ((1 - theo_true_positive) * PL1)
    theo_min_error_round = round(theo_min_error, 3)
    plt.plot(theo_false_positive, theo_true_positive, 'D', label=f'Theoretical Minimum: {theo_min_error_round}')
    plt.xlabel('False Detection')
    plt.ylabel('Correct Detection')
    plt.legend()
    plt.show()





