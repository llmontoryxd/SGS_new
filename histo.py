import numpy as np

from trad import cont_to_indicator_arr


def gaussian(x, mu, sigma):
    a = 1/(sigma*np.sqrt(2*np.pi))
    b = np.exp(-0.5*(((x-mu)/sigma)**2))
    return a*b


def histo(Rest, means, std):
    one_rest = np.sort(Rest[:, :].flatten())

    return one_rest, gaussian(one_rest, means, std)


def histo_U(U):
    U = np.sort(U.flatten())
    return U, gaussian(U, 0, 1)


def qq_plot(Rest, means, std, dist='gaussian', cat_threshold=0):
    one_rest = np.sort(Rest[:, :].flatten())
    norm_one_rest = np.array((one_rest - means)/std)
    np.random.seed(6)
    match dist:
        case 'gaussian':
            random_dist = np.random.normal(0, 1, size=(len(norm_one_rest)))
        case 'category':
            random_dist = np.random.uniform(low=0.0, high=1.0, size=(len(norm_one_rest)))
            random_dist = cont_to_indicator_arr(cat_threshold, random_dist)

    sorted_random_dist = np.sort(random_dist)
    slope, intercept = np.polyfit(sorted_random_dist, norm_one_rest, 1)
    lin_reg_func = intercept + slope*sorted_random_dist

    return sorted_random_dist, norm_one_rest, lin_reg_func