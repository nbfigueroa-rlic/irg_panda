import math, sys
from matplotlib import pyplot as plt
import numpy as np
# import svm_2d_env
# import rbf_2d_env

from sklearn.svm import SVR, SVC

# sys.path.append("environment_generation/")
from obstacles import GammaCircle2D, GammaRectangle2D, GammaCross2D
# import sample_environment
import learn_gamma_fn

from numpy import random as np_random

epsilon = sys.float_info.epsilon

# def null_space_bases(n):
#     '''construct a set of d-1 basis vectors orthogonal to the given d-dim vector n'''
#     d = len(n)
#     es = []
#     for i in range(1, d):
#         e = [-n[i]]
#         for j in range(1, d):
#             if j == i:
#                 e.append(n[0])
#             else:
#                 e.append(0)
#         e = np.stack(e)
#         es.append(e)
#     return es

def null_space_bases(n):
    '''construct a set of d-1 basis vectors orthogonal to the given d-dim vector n'''
    d = len(n)
    es = []    
    # Random vector
    x = np.random.rand(d)    
    # Make it orthogonal to n
    x -= x.dot(n) * n / np.linalg.norm(n)**2
    # normalize it
    x /= np.linalg.norm(x)  
    es.append(x)
    # print(np.dot(n,x))
    if np.dot(n,x) > 1e-10:
        raise AssertionError()

    # if 3d make cross product with x
    if d == 3:        
        y = np.cross(x,n)
        es.append(y)
        # print(np.dot(n,y), np.dot(x,y))
        if np.dot(n,y) > 1e-10:
            raise AssertionError()
        if np.dot(x,y) > 1e-10:
            raise AssertionError()
    return es


def modulation_single_HBS(x, gamma, verbose = False):
    assert len(x.shape) == 1, 'x is not 1-dimensional?'
    d = x.shape[0]
    n = gamma.grad(x)
    # print("Gamma Grad", n)
    es = null_space_bases(n)
    # print("Null space bases", es)

    if verbose:
        print ("x", x)
        print ("nearest point", gamma.nearest_boundary_pt(x))
     
    # if isinstance(gamma.center, (np.ndarray, np.generic)):
    #     r = x - gamma.center        
    elif isinstance(gamma.ref_point, (np.ndarray, np.generic)):
        r = x - gamma.ref_point        
    else:
        r = x - gamma.nearest_boundary_pt(x)

    # print("r", r)
    bases = [r] + es
    # print("bases", bases)
    E = np.stack(bases).T
    # print("E", E)
    E = E / np.linalg.norm(E, axis=0)
    # print("E_norm", E)
    inv_gamma = 1 / gamma(x)
    # print("inv_gamma", inv_gamma)
    lambdas = np.stack([1 - inv_gamma] + [1 + inv_gamma] * (d-1))
    # print("lambdas", lambdas)
    D = np.diag(lambdas)
    # print("D", D)
    invE = np.linalg.inv(E)
    # print("invE", invE)
    return np.matmul(np.matmul(E, D), invE)

def modulation_single_HBS_learned(x, normal_vec, gamma_pt, obstacle_reference_points):
    x = np.array(x)
    assert len(x.shape) == 1, 'x is not 1-dimensional?'
    d = x.shape[0]
    n = normal_vec
    es = null_space_bases(n)
    r =  x - obstacle_reference_points[0]
    bases = [r] + es
    E = np.stack(bases).T
    E = E / np.linalg.norm(E, axis=0)
    inv_gamma = 1 / gamma_pt
    lambdas = np.stack([1 - inv_gamma] + [1 + inv_gamma] * (d-1))
    D = np.diag(lambdas)
    invE = np.linalg.inv(E)
    return np.matmul(np.matmul(E, D), invE)


def modulation_HBS(x, orig_ds, gammas):
    '''
    gammas is a list of k Gamma objects
    '''
    # calculate weight

    # print("x:", x)
    # print("orig_ds:", orig_ds)

    gamma_vals = np.stack([gamma(x) for gamma in gammas])
    print("Gamma vals:", gamma_vals)

    for i in range(gamma_vals.shape[0]):
        if gamma_vals[i] < 1:
            print("COLLIDED.. slipping now..")
            gamma_vals[i] = 1         
    
    ms = np.log(gamma_vals - 1 + epsilon)
    logprod = ms.sum()
    bs = np.exp(logprod - ms)
    weights = bs / bs.sum()

    # print("weights:", weights)

    # calculate modulated dynamical system
    x_dot_mods = []
    for gamma in gammas:
        M = modulation_single_HBS(x, gamma)
        x_dot_mods.append( np.matmul(M, orig_ds.reshape(-1, 1)).flatten() )

    # calculate weighted average of magnitude
    x_dot_mags = np.stack([np.linalg.norm(d) for d in x_dot_mods])
    avg_mag = np.dot(weights, x_dot_mags)
    # print("x dot mags:", x_dot_mags)

    # calculate kappa-space dynamical system and weighted average
    kappas = []
    es = null_space_bases(orig_ds)
    bases = [orig_ds] + es
    R = np.stack(bases).T
    R = R / np.linalg.norm(R, axis=0)

    for x_dot_mod in x_dot_mods:
        n_x_dot_mod = x_dot_mod / np.linalg.norm(x_dot_mod)
        # cob stands for change-of-basis
        n_x_dot_mod_cob = np.matmul(R.T, n_x_dot_mod.reshape(-1, 1)).flatten()
        n_x_dot_mod_cob = n_x_dot_mod_cob / 1.001
        assert -1-1e-5 <= n_x_dot_mod_cob[0] <= 1+1e-5, \
            'n_x_dot_mod_cob[0] = %0.2f?'%n_x_dot_mod_cob[0]
        if n_x_dot_mod_cob[0] > 1:
            acos = np.arccos(n_x_dot_mod_cob[0] - 1e-5)
        elif n_x_dot_mod_cob[0] < -1:
            acos = np.arccos(n_x_dot_mod_cob[0] + 1e-5)
        else:
            acos = np.arccos(n_x_dot_mod_cob[0])
        if np.linalg.norm(n_x_dot_mod_cob[1:]) == 0:
            kappa = acos * n_x_dot_mod_cob[1:] * 0
        else:
            kappa = acos * n_x_dot_mod_cob[1:] / np.linalg.norm(n_x_dot_mod_cob[1:])
        kappas.append(kappa)
    kappas = np.stack(kappas).T
    # matrix-vector multiplication as a weighted sum of columns
    avg_kappa = np.matmul(kappas, weights.reshape(-1, 1)).flatten()

    # map back to task space
    norm = np.linalg.norm(avg_kappa)
    if norm != 0:
        avg_ds_dir = np.concatenate([np.expand_dims(np.cos(norm), 0), avg_kappa * np.sin(norm) / norm])
    else:
        avg_ds_dir = np.concatenate([np.expand_dims(np.cos(norm), 0), avg_kappa])
    avg_ds_dir = np.matmul(R, avg_ds_dir.reshape(-1, 1)).flatten()
    return avg_mag * avg_ds_dir

def modulation_HBS_learned(query_pt, orig_ds, gamma_vals, normal_vecs, obstacle_reference_points):
    '''
    '''
    gamma_pt = gamma_vals[query_pt[0], query_pt[1]]
    normal_vec = normal_vecs[:,query_pt[0],query_pt[1]]
    # calculate modulated dynamical system
    M = modulation_single_HBS_learned(x=query_pt, normal_vec=normal_vec, gamma_pt=gamma_pt, obstacle_reference_points=obstacle_reference_points)
    x_dot_mod = np.matmul(M, orig_ds.reshape(-1, 1)).flatten()
    # calculate average magnitude
    avg_mag = np.linalg.norm(x_dot_mod)

    # calculate kappa-space dynamical system
    es = null_space_bases(orig_ds)
    bases = [orig_ds] + es
    R = np.stack(bases).T
    R = R / np.linalg.norm(R, axis=0)

    n_x_dot_mod = x_dot_mod / np.linalg.norm(x_dot_mod)
    # cob stands for change-of-basis
    n_x_dot_mod_cob = np.matmul(R.T, n_x_dot_mod.reshape(-1, 1)).flatten()
    n_x_dot_mod_cob = n_x_dot_mod_cob / 1.001
    assert -1-1e-5 <= n_x_dot_mod_cob[0] <= 1+1e-5, \
        'n_x_dot_mod_cob[0] = %0.2f?'%n_x_dot_mod_cob[0]
    if n_x_dot_mod_cob[0] > 1:
        acos = np.arccos(n_x_dot_mod_cob[0] - 1e-5)
    elif n_x_dot_mod_cob[0] < -1:
        acos = np.arccos(n_x_dot_mod_cob[0] + 1e-5)
    else:
        acos = np.arccos(n_x_dot_mod_cob[0])
    if np.linalg.norm(n_x_dot_mod_cob[1:]) == 0:
        kappa = acos * n_x_dot_mod_cob[1:] * 0
    else:
        kappa = acos * n_x_dot_mod_cob[1:] / np.linalg.norm(n_x_dot_mod_cob[1:])

    norm = np.linalg.norm(kappa)
    if norm != 0:
        avg_ds_dir = np.concatenate([np.expand_dims(np.cos(norm), 0), kappa * np.sin(norm) / norm])
    else:
        avg_ds_dir = np.concatenate([np.expand_dims(np.cos(norm), 0), kappa])
    avg_ds_dir = np.matmul(R, avg_ds_dir.reshape(-1, 1)).flatten()
    return avg_mag * avg_ds_dir

def linear_controller(x, x_target, max_norm=0.1):
    x_dot = x_target - x
    n = np.linalg.norm(x_dot)
    if n < max_norm:
        return x_dot
    else:
        return x_dot / n * max_norm

def forward_integrate_HBS(x_initial, x_target, gammas, dt, N):
    '''
    forward integration of the HBS modulation controller starting from x_initial,
    toward x_target, with obstacles given as a list of gamma functions.
    integration interval is dt, and N integration steps are performed.
    return an (N+1) x d tensor for x trajectory and an N x d tensor for x_dot trajectory.
    '''
    x_traj = []
    x_traj.append(x_initial)
    x_dot_traj = []
    x_cur = x_initial
    for i in range(N):
        orig_ds = linear_controller(x_cur, x_target)
        x_dot = modulation_HBS(x_cur, orig_ds, gammas)
        x_dot_traj.append(x_dot)
        x_cur = x_cur + x_dot * dt
        x_traj.append(x_cur)
    return np.stack(x_traj), np.stack(x_dot_traj)

def behavior_length(x_traj):
    '''calculate the total length of the trajectory'''
    diffs = x_traj[1:] - x_traj[:-1]
    dists = np.linalg.norm(diffs, axis=1, ord=2)
    return dists.sum()

def test_HBS_fixed_obs():
    '''demo of the HBS approach with multiple obstacles'''
    x_target = np.array([0.9, 0.8])
    gamma1 = GammaCircle2D(np.array(0.15), np.array([0.2, 0.8]))
    gamma2 = GammaRectangle2D(np.array(0.2), np.array(0.3), np.array([0.6, 0.7]))
    gamma3 = GammaCross2D(np.array(0.1), np.array(0.15), np.array([0.7, 0.3]))
    gammas = [gamma1, gamma2, gamma3]
    plt.figure()
    for i in np.linspace(0, 1, 30):
        for j in np.linspace(0, 1, 30):
            x = np.array([i, j])
            if min([gamma(x) for gamma in gammas]) < 1:
                continue
            orig_ds = linear_controller(x, x_target)
            modulated_x_dot = modulation_HBS(x, orig_ds, gammas) * 0.15
            plt.arrow(i, j, modulated_x_dot[0], modulated_x_dot[1],
                head_width=0.008, head_length=0.01)
    for gamma in gammas:
        gamma.draw()
    plt.axis([0, 1, 0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([x_target[0]], [x_target[1]], 'r*')
    # plt.savefig('../images/vector_field_HBS.png', bbox_inches='tight')
    plt.show()


def rand_target_loc(np_random):
    '''
    generate random target location
    '''
    x = np_random.uniform(low=0.05, high=0.5)
    if np_random.randint(0, 2) == 0:
        x = -x
    y = np_random.uniform(low=-0.3, high=0.2)
    z = np_random.uniform(low=0.65, high=1.0)
    return x, z



def test_HBS_fixed_obs_table():
    '''demo of the HBS approach with multiple obstacles'''
    # x_target = np.array([-0.4, 0.8])
    x_target = rand_target_loc(np_random)
    print("x_target:", x_target)    
    gamma1   = GammaRectangle2D(np.array(1.6),  np.array(0.075), np.array([0.0, 0.6]))
    gamma2   = GammaRectangle2D(np.array(0.9),  np.array(0.05),  np.array([0.0, 1.0]))
    gamma3   = GammaRectangle2D(np.array(0.05), np.array(0.4),   np.array([0.0, 0.8]))

    # gamma3 = GammaCross2D(np.array(0.1), np.array(0.15), np.array([0.7, 0.3]))
    gammas = [gamma1, gamma2, gamma3]
    plt.figure()
    for i in np.linspace(-0.8, 0.8, 50):
        for j in np.linspace(0.55, 1.1, 50):                    
            x = np.array([i, j])
            if min([gamma(x) for gamma in gammas]) < 1:
                continue
            orig_ds = linear_controller(x, x_target)
            # modulated_x_dot = modulation_HBS(x, orig_ds, gammas) * 0.15
            modulated_x_dot = modulation_HBS(x, orig_ds, gammas) * 0.05
            # modulated_x_dot = modulated_x_dot / max(1, np.linalg.norm(modulated_x_dot)) * 0.5
            plt.arrow(i, j, modulated_x_dot[0], modulated_x_dot[1],
                head_width=0.008, head_length=0.01)

    for gamma in gammas:
        gamma.draw()
    # plt.axis([0, 1, 0, 1])
    plt.axis([-0.8, 0.8, 0.55, 1.1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([x_target[0]], [x_target[1]], 'r*')
    # plt.savefig('../images/vector_field_HBS.png', bbox_inches='tight')
    plt.show()


def test_HBS_svm_env():
    '''demo of the HBS approach with SVM environment'''

    x_target = np.array([0.9, 0.8])
    # gamma = svm_2d_env.Environment()
    gamma = svm_2d_env.SVMEnvironment()
    plt.figure()
    for i in np.linspace(-1, 1, 40):
        for j in np.linspace(-1, 1, 40):
            x = np.array([i, j])
            if gamma(x) < 1:
                continue
            orig_ds = linear_controller(x, x_target)
            modulated_x_dot = modulation_HBS(x, orig_ds, [gamma]) # * 0.15
            print("modulated x-dot Multi:", modulated_x_dot)

            M = modulation_single_HBS(x, gamma)
            modulated_x_dot = np.matmul(M, orig_ds.reshape(-1, 1)).flatten() 
            print("modulated x-dot Single:", modulated_x_dot)

            modulated_x_dot = modulated_x_dot / max(1, np.linalg.norm(modulated_x_dot)) * 0.1
            plt.arrow(i, j, modulated_x_dot[0], modulated_x_dot[1],
                head_width=0.008, head_length=0.01)
    gamma.draw()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([x_target[0]], [x_target[1]], 'r*')
    # plt.savefig('../images/vector_field_HBS.png', bbox_inches='tight')
    plt.show()

def test_HBS_rbf_env():
    '''demo of the HBS approach with RBF environment'''
    x_target = np.array([0.9, 0.8])
    gamma = rbf_2d_env.RBFEnvironment()

    plt.figure()
    for i in np.linspace(-1, 1, 40):
        for j in np.linspace(-1, 1, 40):
            x = np.array([i, j])
            if gamma(x) < 1:
                continue
            orig_ds = linear_controller(x, x_target)
            modulated_x_dot = modulation_HBS(x, orig_ds, [gamma]) # * 0.15
            print("modulated x-dot Multi:", modulated_x_dot)

            M = modulation_single_HBS(x, gamma)
            modulated_x_dot = np.matmul(M, orig_ds.reshape(-1, 1)).flatten() 
            print("modulated x-dot Single:", modulated_x_dot)


            modulated_x_dot = modulated_x_dot / max(1, np.linalg.norm(modulated_x_dot)) * 0.15
            plt.arrow(i, j, modulated_x_dot[0], modulated_x_dot[1],
                head_width=0.008, head_length=0.01)
    gamma.draw()
    plt.axis([-1, 1, -1, 1])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([x_target[0]], [x_target[1]], 'r*')
    # plt.savefig('../images/vector_field_HBS.png', bbox_inches='tight')
    plt.show()

def test_HBS_learned_obs():
    """
    """
    desc_location = "environment_descriptions/tmp.txt"
    n_obs, start, goal, obstacle_descriptions = sample_environment.sample_environment()
    # sample_environment.draw_environment(start, goal, [0,50], [0,50], obstacle_descriptions, desc_location = desc_location)

    # scale from [-1,1] to our interval [0,50]
    start = np.add(start,1) * 25
    goal = np.add(goal,1) * 25

    X, Y = learn_gamma_fn.read_data(desc_location)
    learned_obstacles = learn_gamma_fn.create_obstacles_from_data(data=X, label=Y, plot_raw_data=False)

    obstacle_reference_points = []
    try:
        for i in range(n_obs):
            obstacle_reference_points.append(learned_obstacles[i]["reference_point"])
        print("Obstacle Reference Points", obstacle_reference_points)
    except:
        print ("SVM found too few obstacles (compared to ground truth)")

    classifier = learned_obstacles['classifier']
    max_dist = learned_obstacles['max_dist']

    xx, yy = np.meshgrid(np.arange(0, 50, 1), np.arange(0, 50, 1))
    position = np.c_[xx.ravel(), yy.ravel()].T

    gamma_vals = learn_gamma_fn.get_gamma(position, classifier, max_dist)
    # normal_vecs = learn_gamma_fn.get_normal_direction(position, classifier, max_dist)
    normal_vecs = learn_gamma_fn.compute_derivatives(position, learned_obstacles)

    fig, ax = learn_gamma_fn.draw_contour_map(classifier, max_dist, gamma_value=True, normal_vecs=normal_vecs, show_vecs=False, show_plot=False)

    gamma_vals = gamma_vals.reshape(xx.shape).T
    normal_vecs = np.swapaxes(normal_vecs.reshape(2, xx.shape[0], xx.shape[1]), 1, 2)
    modulation = np.empty(normal_vecs.shape)

    for i in range(len(xx)):
        for j in range(len(yy)):
            x = [i,j]
            if gamma_vals[i][j] < 1:
                continue
            orig_ds = linear_controller(x, goal)
            modulated_x_dot = modulation_HBS_learned(query_pt=x,
                                                     orig_ds=orig_ds,
                                                     gamma_vals=gamma_vals,
                                                     normal_vecs=normal_vecs,
                                                     obstacle_reference_points=obstacle_reference_points) * 0.15
            modulation[:,i,j] = modulated_x_dot
            plt.arrow(i,j, modulation[0,i,j], modulation[1,i,j], head_width=0.15, head_length=0.8)


    print (modulation[:,int(obstacle_reference_points[0][0]), int(obstacle_reference_points[0][1])])
    modulation.reshape(2,50*50)


    # ax.quiver(xx, yy, modulation[0,:], modulation[1,:])
    # print (modulation.shape)

    #TODO: FIX
    # ax.quiver(xx, yy, modulation[0,:,:], modulation[1,:,:])
    # for i in range(0,50):
    #     for j in range(0,50):
    #         if gamma_vals[i][j] < 1:
    #             continue
    #         plt.arrow(i,j, modulation[0,i,j], modulation[1,i,j], head_width=0.15, head_length=0.8)

    # plt.axis([-10, 60, 60, -10])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([goal[0]], [goal[1]], 'y*',markersize=20)
    # ax.imshow(origin='lower')
    plt.show()

if __name__ == '__main__':
    # test_HBS_learned_obs()
    # test_HBS_fixed_obs()
    # test_HBS_svm_env()
    # test_HBS_rbf_env()

    # New tests
    test_HBS_fixed_obs_table()
