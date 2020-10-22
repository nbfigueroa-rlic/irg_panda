from sklearn import svm
import csv, sys, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def kernel_first_derivative(query_point, support_vector, gamma_svm, der_wrt):
    """
    Cite: https://github.com/nbfigueroa/SVMGrad/blob/master/matlab/kernel/getKernelFirstDerivative.m

    TODO: this function (or compute_derivatives) has a problem; results are incorrect.
    """
    diff = np.subtract(query_point, support_vector)
    if (der_wrt == 1):
        der_val = gamma_svm * np.exp(-gamma_svm*math.pow(np.linalg.norm(diff),2))*diff
        # der_val = -2 * np.exp(-((gamma_svm * diff).T * diff)) * (gamma_svm * diff)
    else:
        der_val = gamma_svm * np.exp(-gamma_svm*math.pow(np.linalg.norm(diff),2))*np.subtract(support_vector,query_point)
        # der_val = -2 * np.exp(-((gamma_svm * diff).T * diff)) * (gamma_svm * np.subtract(support_vector,query_point))
    return der_val

def compute_derivatives(positions, learned_obstacles):
    """
    Cite: https://github.com/nbfigueroa/SVMGrad/blob/master/matlab/classifier/calculateGammaDerivative.m

    TODO: this function (or kernel_first_derivative) has a problem; results are incorrect.
    """
    normal_vecs = np.zeros((positions.shape))
    alphas = learned_obstacles["classifier"].dual_coef_[0]
    gamma_svm = learned_obstacles["gamma_svm"]
    support_vectors = learned_obstacles["classifier"].support_vectors_

    for idx in range(len(normal_vecs[0,:])):
        query_point = positions[:,idx]
        normal = np.zeros((query_point.shape))

        for sv_idx in range(len(support_vectors)):
            normal += alphas[sv_idx] * kernel_first_derivative(query_point, support_vectors[sv_idx], gamma_svm=gamma_svm, der_wrt=1)
        normal_vecs[:,idx] = normal
    return normal_vecs

def get_normal_direction(position, classifier, max_dist, normalize=True, delta_dist=1.e-5, dimension=2):
    '''
    Numerical differentiation to of Gamma to get normal direction.
    Cite: https://github.com/epfl-lasa/dynamic_obstacle_avoidance_linear
    '''
    pos_shape = position.shape
    # fairly certain this next line doesn't do anything, but it's in Lukas's code
    positions = position.reshape(dimension, -1)

    normals = np.zeros((positions.shape))

    for dd in range(dimension):
        pos_low, pos_high = np.copy(positions), np.copy(positions)
        pos_high[dd, :] = pos_high[dd, :] + delta_dist
        pos_low[dd, :] = pos_low[dd, :] - delta_dist

        normals[dd, :] = (get_gamma(pos_high, classifier, max_dist) - \
                          get_gamma(pos_low, classifier, max_dist))/2*delta_dist

    if normalize:
        mag_normals = np.linalg.norm(normals, axis=0)
        nonzero_ind = mag_normals>0

        if any(nonzero_ind):
            normals[:, nonzero_ind] = normals[:, nonzero_ind] / mag_normals[nonzero_ind]

    # return (-1)*normals # due to gradent definition
    return normals

def learn_obstacles_from_data(data_obs, data_free, C_svm=1000):
    """
    Cite: https://github.com/epfl-lasa/dynamic_obstacle_avoidance_linear
    """
    data = np.hstack((data_free, data_obs))
    label = np.hstack(( np.zeros(data_free.shape[1]), np.ones(data_obs.shape[1]) ))

    n_features = data.T.shape[1]
    gamma_svm = 1 / (n_features * data.T.var())

    classifier = svm.SVC(kernel='rbf', gamma=gamma_svm, C=C_svm).fit(data.T, label)
    print('Number of support vectors / data points')
    print('Free space: ({} / {}) --- Obstacle ({} / {})'.format(
        classifier.n_support_[0], data_free.shape[1],
        classifier.n_support_[1], data_obs.shape[1]))

    dist = np.linalg.norm(data_obs-np.tile((0,0), (data_obs.shape[1], 1)).T, axis=0)
    max_dist = np.max(dist)

    return classifier, max_dist, gamma_svm, C_svm

def get_gamma(position, classifier, max_dist, reference_point=(0,0)):
    """
    Cite: https://github.com/epfl-lasa/dynamic_obstacle_avoidance_linear
    """
    pos_shape = position.shape

    score = classifier.decision_function(np.c_[position[0].T, position[1].T])
    dist = np.linalg.norm(position - np.tile(reference_point, (position.shape[1], 1)).T, axis=0)

    outer_ref_dist = max_dist*2
    # dist = np.clip(dist, max_dist, outer_ref_dist)

    ind_noninf = outer_ref_dist > dist
    distance_score = (outer_ref_dist-max_dist)/(outer_ref_dist-dist[ind_noninf])

    max_float = sys.float_info.max
    max_float = 1e12
    gamma = np.zeros(dist.shape)
    gamma[ind_noninf] = (-score[ind_noninf] + 1) * distance_score
    gamma[~ind_noninf] = max_float

    if len(pos_shape)==1:
        gamma = gamma[0]
    return gamma


def create_obstacles_from_data(data, label, cluster_eps=10, cluster_min_samples=10, label_free=0, label_obstacle=1, plot_raw_data=False):
    """
    Cite: https://github.com/epfl-lasa/dynamic_obstacle_avoidance_linear

    Parameters:
        data : list of form [[x1 x2 x3 ...], [y1 y2 y3 ...]]
        label : list of form [l1 l2 l3 ...]
        cluster_eps : int, default 10. input to DBSCAN.
        cluster_min_samples : int, default 10. input to DBSCAN
        label_free : int, default 0. The label denoting free space.
        label_obstacle : int, default 1. The label denoting obstacles.
        plot_raw_data : boolean, whether to draw raw data

    Returns:
        obstacles : dictionary of form {"classifier": SVM, "max_dist": (float), "obstacle_1": (float, float), "obstacle2": (float, float), ...}

    """
    data_obs = data[:, label==label_obstacle]
    data_free = data[:, label==label_free]

    if plot_raw_data:
        plt.figure(figsize=(6, 6))
        plt.plot(data_free[0, :], data_free[1, :], '.', color='#57B5E5', label='No Collision')
        plt.plot(data_obs[0, :], data_obs[1, :], '.', color='#833939', label='Collision')
        plt.axis('equal')
        plt.title("Raw Data")
        plt.legend()

        plt.xlim([np.min(data[0, :]), np.max(data[0, :])])
        plt.ylim([np.min(data[1, :]), np.max(data[1, :])])
        plt.pause(0.01)
        plt.show()

    # Alternatively - we can use the centers of the generated polygons.
    clusters = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit(data_obs.T)
    cluster_labels, obs_index = np.unique(clusters.labels_, return_index=True)

    n_obstacles = np.sum(cluster_labels>=0)
    obs_points = []
    learned_obstacles = {}

    classifier, max_dist, gamma_svm, c_svm = learn_obstacles_from_data(data_obs=data_obs, data_free=data_free)

    learned_obstacles["classifier"] = classifier
    learned_obstacles["max_dist"] = max_dist
    learned_obstacles["gamma_svm"] = gamma_svm
    learned_obstacles["c_svm"] = c_svm


    for oo in range(n_obstacles):
        ind_clusters = (clusters.labels_==oo)
        obs_points.append(data_obs[:, ind_clusters])
        mean_position = np.mean(obs_points[-1], axis=1)
        learned_obstacles[oo] = {"reference_point": mean_position}

    return (learned_obstacles)


def draw_contour_map(classifier, max_dist, fig=None, ax=None, show_contour=True, gamma_value=False, normal_vecs = None, show_vecs=True, show_plot = True):
    """
    Cite: https://github.com/epfl-lasa/dynamic_obstacle_avoidance_linear
    """
    xx, yy = np.meshgrid(np.arange(0, 50, 1), np.arange(0, 50, 1))

    if ax is None or fig is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 6)

    if gamma_value:
        predict_score = get_gamma(np.c_[xx.ravel(), yy.ravel()].T, classifier, max_dist)
        predict_score = predict_score - 1 # Subtract 1 to have differentiation boundary at 1
        plt.title("$\Gamma$-Score")
    else:
        predict_score = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        plt.title("SVM Score")
    predict_score = predict_score.reshape(xx.shape)
    levels = np.array([0])

    cs0 = ax.contour(xx, yy, predict_score, levels, origin='lower', colors='k', linewidths=2)
    if show_contour:

        cs = ax.contourf(xx, yy, predict_score, np.arange(-16, 16, 2),
                         cmap=plt.cm.coolwarm, extend='both', alpha=0.8)

        cbar = fig.colorbar(cs)
        cbar.add_lines(cs0)

        if show_vecs:
            ax.quiver(xx, yy, normal_vecs[0, :], normal_vecs[1, :])


    else:
        cmap = colors.ListedColormap(['#000000', '#A86161'])
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0.05, 0.7, cmap.N)
        my_cmap = ListedColormap(my_cmap)
        bounds=[-1,0,1]
        norm = colors.BoundaryNorm(bounds, my_cmap.N)

        alphas = 0.5

        cs = ax.contourf(xx, yy, predict_score, origin='lower', cmap=my_cmap, norm=norm)

        reference_point = (0,0)
        ax.plot(reference_point[0],reference_point[1], 'k+', linewidth=18, markeredgewidth=4, markersize=13)
    if show_plot:
        plt.show()
    return (fig, ax)

def read_data (file_location):
    """
    Read data from a CSV

    Parameters:
        file_location : string

    Returns:
        X : numpy array, list of lists, [[x1, x2, x3, ...], [y1, y2, y3, ...]]
        Y : numpy array, list of floats [label1, label 2, ...]
    """
    X = [[],[]]
    Y = []

    with open(file_location, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        header = True
        for row in csvreader:
            # ignore header
            if header:
                header = False
                continue

            X[0].append(int(row[1]))
            X[1].append(int(row[2]))
            Y.append(float(row[0]))

        return (np.array(X), np.array(Y))

if __name__ == '__main__':
    X, Y = read_data("environment_descriptions/tmp.txt")
    learned_obstacles = create_obstacles_from_data(data=X, label=Y, plot_raw_data=True)
    classifier = learned_obstacles['classifier']
    max_dist = learned_obstacles['max_dist']

    xx, yy = np.meshgrid(np.arange(0, 50, 1), np.arange(0, 50, 1))
    positions = np.c_[xx.ravel(), yy.ravel()].T

    normal_vecs = get_normal_direction(positions, classifier, max_dist)
    draw_contour_map(classifier, max_dist, gamma_value=True, normal_vecs=normal_vecs)

    normal_vecs = compute_derivatives(positions, learned_obstacles)
    draw_contour_map(classifier, max_dist, gamma_value=True, normal_vecs=normal_vecs)
