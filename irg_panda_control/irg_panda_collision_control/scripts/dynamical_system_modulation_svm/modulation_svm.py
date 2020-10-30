import math, sys
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVR, SVC
from numpy import random as np_random

# sys.path.append("environment_generation/")
# from obstacles import GammaCircle2D, GammaRectangle2D, GammaCross2D
# import sample_environment

import learn_gamma_fn
from modulation_utils import *
# import rbf_2d_env_svm

epsilon = sys.float_info.epsilon

#######################
## Common Functions  ##
#######################
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


def linear_controller(x, x_target, max_norm=0.1):
    x_dot = x_target - x
    n = np.linalg.norm(x_dot)
    if n < max_norm:
        return x_dot
    else:
        return x_dot / n * max_norm

def behavior_length(x_traj):
    '''calculate the total length of the trajectory'''
    diffs = x_traj[1:] - x_traj[:-1]
    dists = np.linalg.norm(diffs, axis=1, ord=2)
    return dists.sum()

########################################################################################################################
## For use with the parametrized gamma functions (circles, rectangles, cross) as well as the RBF defined environments ##
########################################################################################################################

def modulation_single_HBS(x, orig_ds, gamma, verbose = False, ref_adapt = True):
    '''
        Compute modulation matrix for a single obstacle described with a gamma function
        and unique reference point
    '''
    x = np.array(x)
    assert len(x.shape) == 1, 'x is not 1-dimensional?'
    normal_vec      = gamma.grad(x)
    reference_point = gamma.center
    gamma_pt        = gamma(x)

    M  = modulation_singleGamma_HBS(x=x, orig_ds=orig_ds, normal_vec=normal_vec, gamma_pt = gamma_pt, reference_point = reference_point, ref_adapt = ref_adapt)
    return M


def modulation_HBS(x, orig_ds, gammas):
    '''
    gammas is a list of k Gamma objects
    '''

    gamma_vals = np.stack([gamma(x) for gamma in gammas])

    if (gamma_vals.shape[0] == 1 and gamma_vals < 1.0) or (gamma_vals.shape[0] > 1 and any(gamma_vals< 1.0) ):
        return np.zeros(orig_ds.shape)

    if (gamma_vals.shape[0] == 1 and gamma_vals > 1e9) or (gamma_vals.shape[0] > 1 and any(gamma_vals > 1e9)):
        return orig_ds

    ms = np.log(gamma_vals - 1 + epsilon)
    logprod = ms.sum()
    bs = np.exp(logprod - ms)
    weights = bs / bs.sum()


    # calculate modulated dynamical systems
    x_dot_mods = []
    for gamma in gammas:
        x_dot_mod = modulation_singleGamma_HBS_multiRef(query_pt=x, orig_ds=orig_ds, gamma_query=gamma(x),
            normal_vec_query=gamma.grad(x), obstacle_reference_points= gamma.center, repulsive_gammaMargin = 0.01)
        x_dot_mods.append(x_dot_mod)
        # The function below does now account for extra modifications on modulated velocity
        # M = modulation_single_HBS(x, gamma)
        # x_dot_mods.append( np.matmul(M, orig_ds.reshape(-1, 1)).flatten() )


    # calculate weighted average of magnitude
    x_dot_mags = np.stack([np.linalg.norm(d) for d in x_dot_mods])
    avg_mag = np.dot(weights, x_dot_mags)

    old_way = True
    if old_way:

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

    else:

        dim = len(x)
        x_dot_mods_normalized = np.zeros((dim, len(gammas)))
        x_dot_mods_np = np.array(x_dot_mods).T

        ind_nonzero = (x_dot_mags>0)
        if np.sum(ind_nonzero):
            x_dot_mods_normalized[:, ind_nonzero] = x_dot_mods_np[:, ind_nonzero]/np.tile(x_dot_mags[ind_nonzero], (dim, 1))
        x_dot_normalized = orig_ds / np.linalg.norm(orig_ds)

        avg_ds_dir = get_directional_weighted_sum(reference_direction=x_dot_normalized,
            directions=x_dot_mods_normalized, weights=weights, total_weight=1)

        x_mod_final = avg_mag*avg_ds_dir.squeeze()

    x_mod_final = avg_mag * avg_ds_dir

    return x_mod_final


def forward_integrate_HBS(x_initial, x_target, gammas, dt, eps, max_N):
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
    for i in range(max_N):
        orig_ds = linear_controller(x_cur, x_target)
        x_dot = modulation_HBS(x_cur, orig_ds, gammas)
        x_dot_traj.append(x_dot)
        x_cur = x_cur + x_dot * dt
        if np.linalg.norm(x_cur - x_target) < eps:
            print("Attractor Reached")
            break
        x_traj.append(x_cur)
    return np.stack(x_traj), np.stack(x_dot_traj)



def forward_integrate_singleGamma_HBS(x_initial, x_target, learned_gamma, dt, eps, max_N):
    '''
    forward integration of the HBS modulation controller starting from x_initial,
    toward x_target, with obstacles given as a list of gamma functions.
    integration interval is dt, and N integration steps are performed.
    return an (N+1) x d tensor for x trajectory and an N x d tensor for x_dot trajectory.
    '''

    # Parse Gamma
    classifier        = learned_gamma['classifier']
    max_dist          = learned_gamma['max_dist']
    reference_points  = learned_gamma['reference_points']
    dim = len(x_target)
    print(dim)
    x_traj = []
    x_traj.append(x_initial)
    x_dot_traj = []
    x_cur = x_initial
    print("Before Integration")
    for i in range(max_N):
        print(x_cur)
        print(x_target)
        gamma_val  = learn_gamma_fn.get_gamma(x_cur, classifier, max_dist, reference_points, dimension=dim)
        print(gamma_val)
        normal_vec = learn_gamma_fn.get_normal_direction(x_cur, classifier, reference_points, max_dist, dimension=dim)        
        print(normal_vec)
        orig_ds    = linear_controller(x_cur, x_target)
        print(orig_ds)
        x_dot      = modulation_singleGamma_HBS_multiRef(query_pt=x_cur, orig_ds=orig_ds, gamma_query=gamma_val,
                            normal_vec_query=normal_vec.reshape(dim), obstacle_reference_points=reference_points, repulsive_gammaMargin=0.01)
        x_dot      = x_dot/np.linalg.norm(x_dot + epsilon) * 0.03 
        x_dot_traj.append(x_dot)
        x_cur = x_cur + x_dot * dt
        if np.linalg.norm(x_cur - x_target) < eps:
            print("Attractor Reached")
            break
        x_traj.append(x_cur)
    return np.stack(x_traj), np.stack(x_dot_traj)



######################################################################################################################
## For use with non-class defined gamma functions (singleGamma is a single gamma function describing all obstacles) ##
######################################################################################################################
def modulation_singleGamma_HBS(x, orig_ds, normal_vec, gamma_pt, reference_point, ref_adapt = True, tangent_scale_max = 5.0):
    '''
        Compute modulation matrix for a single obstacle described with a gamma function
        and unique reference point
    '''
    x = np.array(x)
    assert len(x.shape) == 1, 'x is not 1-dimensional?'
    d = x.shape[0]
    n = normal_vec

    # Compute the Eigen Bases by adapting the reference direction!
    if ref_adapt:
        E, E_orth = compute_decomposition_matrix(x, normal_vec, reference_point)
    else:
        # Compute the Eigen Bases by NO adaptation of the reference direction!
        es = null_space_bases(n)
        r =  x - reference_point
        bases = [r] + es
        E = np.stack(bases).T
        E = E / np.linalg.norm(E, axis=0)

    # print("gamma", gamma_pt)
    # print("E", E)
    invE = np.linalg.inv(E)

    # Compute Diagonal Matrix
    tangent_scaling = 1
    if gamma_pt <=1:
        inv_gamma = 1
    else:
        inv_gamma       = 1 / gamma_pt
        # Consider TAIL EFFECT!
        tail_angle = np.dot(normal_vec, orig_ds)
        if (tail_angle) < 0:
            # print("Going TOWARDS obstacle!")
            tangent_scaling = max(1, tangent_scale_max - (1-inv_gamma))
            lambdas = np.stack([1 - inv_gamma] + [tangent_scaling*(1 + inv_gamma)] * (d-1))            
        else:
            # print("Going AWAY from obstacle!")
            tangent_scale_max = 1.0
            tangent_scaling   = 1.0
            lambdas = np.stack([1] + [tangent_scaling*(1 + inv_gamma)] * (d-1))



    D = np.diag(lambdas)
    # print("D", D)
    M = np.matmul(np.matmul(E, D), invE)

    if gamma_pt < 1.0:
        M =  np.zeros(M.shape)

    if gamma_pt > 1e9:
        M =  np.identity(d)

    return M


def modulation_singleGamma_HBS_multiRef(query_pt, orig_ds, gamma_query, normal_vec_query, obstacle_reference_points, repulsive_gammaMargin = 0.01, sticky_surface = False):
    '''
        Computes modulated velocity for an environment described by a single gamma function
        and multiple reference points (describing multiple obstacles)
    '''

    # Add: Expand Boundary -- this doesn't work as I hoped
    # if gamma_query < 1.25 and gamma_query > 1.0:
    #     gamma_query = 1.0

    # Add: Contingencies for low gamma (inside obstacle) and high gamma (somewhere far away!)
    if gamma_query < 1.0:
        return np.zeros(orig_ds.shape)

    if gamma_query > 1e9:
        return orig_ds

    try:
        len(obstacle_reference_points.shape)
        reference_point = obstacle_reference_points
        # print("HEEEERE")
    except:
        reference_point = find_closest_reference_point(query_pt, obstacle_reference_points)

    # Add: Move away from center/reference point in case of a collision
    pos_relative     = -(query_pt - reference_point)
    if gamma_query < ( 1 + repulsive_gammaMargin):
        repulsive_power =  5
        repulsive_factor = 5
        repulsive_gamma = (1 + repulsive_gammaMargin)
        repulsive_speed =  ((repulsive_gamma/gamma_query)**repulsive_power-
                               repulsive_gamma)*repulsive_factor
        norm_xt = np.linalg.norm(pos_relative)
        if (norm_xt): # nonzero
            repulsive_velocity = pos_relative/np.linalg.norm(pos_relative) * repulsive_speed
        else:
            repulsive_velocity = np.zeros(dim)
            repulsive_velocity[0] = 1*repulsive_speed
        x_dot_mod = -repulsive_velocity
    else:
        # Calculate real modulated dynamical system
        M = modulation_singleGamma_HBS(x=query_pt, orig_ds = orig_ds, normal_vec=normal_vec_query, gamma_pt=gamma_query,
            reference_point=reference_point)
        x_dot_mod = np.matmul(M, orig_ds.reshape(-1, 1)).flatten()

    # Add: Sticky Surface
    if sticky_surface:
        xd_relative_norm = np.linalg.norm(orig_ds)
        if xd_relative_norm:
            # Limit maximum magnitude
            eigenvalue_magnitude = 1 - 1./abs(gamma_query)**1
            mag = np.linalg.norm(mod_ds)
            mod_ds = mod_ds/mag*xd_relative_norm * eigenvalue_magnitud

    return x_dot_mod


# STATUS: WORKS!!!!
def modulation_multiGamma_HBS_svm(query_pt, orig_ds, learned_gammas, repulsive_gammaMargin = 0.01, sticky_surface = False):
    '''
        Computes modulated velocity for an environment described by a single gamma function
        and multiple reference points (describing multiple obstacles)
    '''

    # Add: Expand Boundary
    # if gamma_query < 1.25 and gamma_query > 1.0:
    #     gamma_query = 1.0

    dim = len(query_pt)
    gamma_vals =[]
    for oo in range(len(learned_gammas)):
        gamma_vals.append(learn_gamma_fn.get_gamma(query_pt.reshape(dim,1), learned_gammas[oo]['classifier'], learned_gammas[oo]['max_dist'], learned_gammas[oo]['reference_point']))


    gamma_vals = np.array(gamma_vals)
    if (len(gamma_vals) == 1 and gamma_vals < 1.0) or (len(gamma_vals) > 1 and any(gamma_vals < 1.0)):
        return np.zeros(orig_ds.shape)

    if (len(gamma_vals) == 1 and gamma_vals > 1e9) or (len(gamma_vals) > 1 and any(gamma_vals > 1e9)):
        return orig_ds

    if len(obstacle_reference_points.shape) > 1:
        reference_point = find_closest_reference_point(query_pt, obstacle_reference_points)
    else:
        reference_point = obstacle_reference_points

    # Add: Move away from center/reference point in case of a collision
    pos_relative     = -(query_pt - reference_point)
    if gamma_query < ( 1 + repulsive_gammaMargin):
        repulsive_power =  5
        repulsive_factor = 5
        repulsive_gamma = (1 + repulsive_gammaMargin)
        repulsive_speed =  ((repulsive_gamma/gamma_query)**repulsive_power-
                               repulsive_gamma)*repulsive_factor
        norm_xt = np.linalg.norm(pos_relative)
        if (norm_xt): # nonzero
            repulsive_velocity = pos_relative/np.linalg.norm(pos_relative) * repulsive_speed
        else:
            repulsive_velocity = np.zeros(dim)
            repulsive_velocity[0] = 1*repulsive_speed
        x_dot_mod = repulsive_velocity
    else:
        # Calculate real modulated dynamical system
        M = modulation_singleGamma_HBS(x=query_pt, orig_ds = orig_ds, normal_vec=normal_vec_query, gamma_pt=gamma_query,
            reference_point=reference_point)
        x_dot_mod = np.matmul(M, orig_ds.reshape(-1, 1)).flatten()


    ms = np.log(gamma_vals - 1 + epsilon)
    logprod = ms.sum()
    bs = np.exp(logprod - ms)
    weights = bs / bs.sum()
    weights = weights.T[0]

    # calculate modulated dynamical systems
    x_dot_mods = []
    for oo in range(len(learned_gammas)):
        normal_vec = learn_gamma_fn.get_normal_direction(query_pt.reshape(dim,1), classifier=learned_gammas[oo]['classifier'],reference_point=learned_gammas[oo]['reference_point'],
            max_dist= learned_gammas[oo]['max_dist'], gamma_svm=learned_gammas[oo]['gamma_svm'])
        x_dot_mod = modulation_singleGamma_HBS_multiRef(query_pt=query_pt, orig_ds=orig_ds, gamma_query=gamma_vals[oo][0],
            normal_vec_query=normal_vec.flatten(), obstacle_reference_points = learned_gammas[oo]['reference_point'], repulsive_gammaMargin = 0.01)
        x_dot_mods.append(x_dot_mod)

    # calculate weighted average of magnitude
    x_dot_mags = np.stack([np.linalg.norm(d) for d in x_dot_mods])
    avg_mag = np.dot(weights, x_dot_mags)

    old_way = True
    if old_way:

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

    else:

        x_dot_mods_normalized = np.zeros((dim, len(learned_gammas)))
        x_dot_mods_np = np.array(x_dot_mods).T

        ind_nonzero = (x_dot_mags>0)
        if np.sum(ind_nonzero):
            x_dot_mods_normalized[:, ind_nonzero] = x_dot_mods_np[:, ind_nonzero]/np.tile(x_dot_mags[ind_nonzero], (dim, 1))
        x_dot_normalized = orig_ds / np.linalg.norm(orig_ds)

        avg_ds_dir = get_directional_weighted_sum(reference_direction=x_dot_normalized,
            directions=x_dot_mods_normalized, weights=weights, total_weight=1)

        x_mod_final = avg_mag*avg_ds_dir.squeeze()

    x_mod_final = avg_mag * avg_ds_dir

def draw_modulated_svmGamma_HBS(x_target, reference_points, gamma_vals, normal_vecs, nb_obstacles, grid_limits_x, grid_limits_y, grid_size, x_initial, learned_gamma, dt,filename='tmp.png', data = []):

    fig, ax1 = plt.subplots()
    ax1.set_xlim(grid_limits_x[0], grid_limits_x[1])
    ax1.set_ylim(grid_limits_y[0], grid_limits_y[1])
    plt.xlabel('$x_1$',fontsize=15)
    plt.ylabel('$x_2$',fontsize=15)
    plt.title('HBS Modulated DS',fontsize=15)
    X, Y = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size))
    V, U = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            x_query    = np.array([X[i,j], Y[i,j]])
            orig_ds    = linear_controller(x_query, x_target)
            mod_x_dot  = modulation_singleGamma_HBS_multiRef(query_pt=x_query, orig_ds=orig_ds, gamma_query=gamma_vals[i,j],
                                normal_vec_query=normal_vecs[:,i,j], obstacle_reference_points=reference_points, repulsive_gammaMargin=0.01)
            x_dot_norm = mod_x_dot/np.linalg.norm(mod_x_dot + epsilon) * 0.20
            U[i,j]     = x_dot_norm[0]
            V[i,j]     = x_dot_norm[1]

    strm      = ax1.streamplot(X, Y, U, V, density = 3.5, linewidth=0.55, color='k')
    levels    = np.array([0, 1])
    cs0       = ax1.contour(X, Y, gamma_vals, levels, origin='lower', colors='k', linewidths=1)
    cs        = ax1.contourf(X, Y, gamma_vals, np.arange(-5, 10, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
    # cs        = ax1.contourf(X, Y, gamma_vals, np.arange(-16, 16, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
    cbar      = plt.colorbar(cs)
    cbar.add_lines(cs0)

    if len(reference_points) > 1:
        for oo in range(nb_obstacles):
            reference_point = reference_points[oo]
            print(reference_point)
            ax1.plot([reference_point[0]], [reference_point[1]], 'k+')
    else: 
        ax1.plot([reference_points[0]], [reference_points[1]], 'k+')       
    plt.gca().set_aspect('equal', adjustable='box')       

    # Integrate trajectories from initial point
    x, x_dot = forward_integrate_singleGamma_HBS(x_initial, x_target, learned_gamma, dt, eps=0.03, max_N = 10000)
    ax1.plot(x.T[0,:], x.T[1,:], 'b.')       

    if data == []:    
        print("Don't plot data")
    else:
        ax1.plot([data[0,:]], [data[1,:]], 'm.')           


    ax1.plot(x_target[0], x_target[1], 'md', markersize=12, lw=2)
    # plt.savefig(filename)
    plt.savefig(filename+".png", dpi=300)
    plt.savefig(filename+".pdf", dpi=300)

# def draw_modulated_multisvmGamma_HBS(x_target, learned_gammas, grid_limits, grid_size, filename='./dynamical_system_modulation_svm/figures/tmp.png'):

#     fig, ax1 = plt.subplots()
#     ax1.set_xlim(grid_limits[0], grid_limits[1])
#     ax1.set_ylim(grid_limits[0], grid_limits[1])
#     plt.xlabel('$x_1$',fontsize=15)
#     plt.ylabel('$x_2$',fontsize=15)
#     plt.title('HBS Modulated DS',fontsize=15)
#     X, Y = np.meshgrid(np.linspace(grid_limits[0], grid_limits[1], grid_size), np.linspace(grid_limits[0], grid_limits[1], grid_size))
#     V, U = np.meshgrid(np.linspace(grid_limits[0], grid_limits[1], grid_size), np.linspace(grid_limits[0], grid_limits[1], grid_size))
#     gamma_vals = np.ndarray(X.shape)
#     for i in range(grid_size):
#         for j in range(grid_size):
#             x_query    = np.array([X[i,j], Y[i,j]])

#             gamma_vals_query =[]
#             for oo in range(len(learned_gammas)):
#                 gamma_vals_query.append(learn_gamma_fn.get_gamma(x_query.reshape(2,1), learned_gammas[oo]['classifier'], learned_gammas[oo]['max_dist'], learned_gammas[oo]['reference_point']))
#             # print("gammas:", gamma_vals_query)
#             gamma_vals[i,j] = min(gamma_vals_query)
#             orig_ds    = linear_controller(x_query, x_target)
#             mod_x_dot  = modulation_multiGamma_HBS_svm(query_pt = x_query, orig_ds = orig_ds, learned_gammas = learned_gammas, repulsive_gammaMargin = 0.01, sticky_surface = False)
#             x_dot_norm = mod_x_dot/np.linalg.norm(mod_x_dot + epsilon) * 0.20
#             U[i,j]     = x_dot_norm[0]
#             V[i,j]     = x_dot_norm[1]

#     strm      = ax1.streamplot(X, Y, U, V, density = 3.5, linewidth=0.55, color='k')
#     levels    = np.array([0, 1])
#     cs0       = ax1.contour(X, Y, gamma_vals, levels, origin='lower', colors='k', linewidths=1)
#     cs        = ax1.contourf(X, Y, gamma_vals, np.arange(-5, 10, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
#     # cs        = ax1.contourf(X, Y, gamma_vals, np.arange(-16, 16, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
#     cbar      = plt.colorbar(cs)
#     cbar.add_lines(cs0)

#     for oo in range(len(learned_gammas)):
#         ax1.plot([learned_gammas[oo]['reference_point'][0]], [learned_gammas[oo]['reference_point'][1]], 'y*')
#     ax1.plot(x_target[0], x_target[1], 'md', markersize=12, lw=2)
#     plt.savefig(filename)



# def draw_modulated_multiGamma_HBS(x_target, gammas, grid_limits, grid_size, filename='./dynamical_system_modulation_svm/figures/tmp.png'):
#     fig, ax1 = plt.subplots()
#     ax1.set_xlim(grid_limits[0], grid_limits[1])
#     ax1.set_ylim(grid_limits[0], grid_limits[1])
#     plt.xlabel('$x_1$',fontsize=15)
#     plt.ylabel('$x_2$',fontsize=15)
#     plt.title('HBS Modulated DS',fontsize=15)

#     X, Y = np.meshgrid(np.linspace(grid_limits[0], grid_limits[1], grid_size), np.linspace(grid_limits[0], grid_limits[1], grid_size))
#     V, U = np.meshgrid(np.linspace(grid_limits[0], grid_limits[1], grid_size), np.linspace(grid_limits[0], grid_limits[1], grid_size))
#     gamma_vals = np.ndarray(X.shape)
#     for i in range(grid_size):
#         for j in range(grid_size):
#             x_query          = np.array([X[i,j], Y[i,j]])
#             gamma_vals[i,j] = min([gamma(x_query) for gamma in gammas])
#             orig_ds         = linear_controller(x_query, x_target)
#             mod_x_dot       = modulation_HBS(x_query, orig_ds, gammas)
#             x_dot_norm      = mod_x_dot/np.linalg.norm(mod_x_dot + epsilon) * 0.20
#             U[i,j]          = x_dot_norm[0]
#             V[i,j]          = x_dot_norm[1]

#     # for gamma in gammas:
#     #     gamma.draw()

#     strm      = ax1.streamplot(X, Y, U, V, density = 4.5, linewidth=0.55, color='k')
#     # strm      = ax1.quiver(X, Y, U, V, linewidth=0.55, color='k')
#     levels    = np.array([0, 1, 1.25])
#     cs0       = ax1.contour(X, Y, gamma_vals, levels, origin='lower', colors='k', linewidths=1)
#     cs        = ax1.contourf(X, Y, gamma_vals, np.arange(-5, 10, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
#     cs        = ax1.contourf(X, Y, gamma_vals, np.arange(-15, 15, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
#     cbar      = plt.colorbar(cs)
#     cbar.add_lines(cs0)
#     ax1.plot(x_target[0], x_target[1], 'md', markersize=12, lw=2)
#     for gamma in gammas:
#         ax1.plot([gamma.center[0]], [gamma.center[1]], 'y*')

#     plt.savefig(filename, bbox_inches='tight')
#     plt.show()


# ########################################################
# ##     Tests to run different modulation scenarios    ##
# ########################################################

# # STATUS: Averaging messes up some of the obstacles modulation
# def test_HBS_fixed_obs():
#     '''demo of the HBS approach with multiple obstacles'''
#     # Defining environment!
#     grid_limits = [0,1]
#     grid_size   = 30
#     x_target = np.array([0.9, 0.8])
#     gamma1 = GammaCircle2D(np.array(0.15), np.array([0.2, 0.8]))
#     gamma2 = GammaRectangle2D(np.array(0.2), np.array(0.3), np.array([0.6, 0.7]))
#     gamma3 = GammaCross2D(np.array(0.1), np.array(0.15), np.array([0.7, 0.3]))
#     gammas = [gamma1, gamma2, gamma3]
#     gammas = [gamma1, gamma2]
#     filename = './dynamical_system_modulation_svm/figures//parametrizedGamma_multiple_modHBS.png'
#     draw_modulated_multiGamma_HBS(x_target, gammas, grid_limits, grid_size, filename)


# # STATUS: NEEDS TO BE TESTED FURTHER!
# def test_HBS_fixed_obs_table():
#     '''demo of the HBS approach with multiple obstacles'''
#     # x_target = np.array([-0.4, 0.8])
#     x_target = rand_target_loc(np_random)
#     print("x_target:", x_target)
#     gamma1   = GammaRectangle2D(np.array(1.6),  np.array(0.075), np.array([0.0, 0.6]))
#     gamma2   = GammaRectangle2D(np.array(0.9),  np.array(0.05),  np.array([0.0, 1.0]))
#     gamma3   = GammaRectangle2D(np.array(0.05), np.array(0.4),   np.array([0.0, 0.8]))

#     # gamma3 = GammaCross2D(np.array(0.1), np.array(0.15), np.array([0.7, 0.3]))
#     gammas = [gamma1, gamma2, gamma3]
#     plt.figure()
#     for i in np.linspace(-0.8, 0.8, 50):
#         for j in np.linspace(0.55, 1.1, 50):
#             x = np.array([i, j])
#             if min([gamma(x) for gamma in gammas]) < 1:
#                 continue
#             orig_ds = linear_controller(x, x_target)
#             # modulated_x_dot = modulation_HBS(x, orig_ds, gammas) * 0.15
#             modulated_x_dot = modulation_HBS(x, orig_ds, gammas) * 0.05
#             # modulated_x_dot = modulated_x_dot / max(1, np.linalg.norm(modulated_x_dot)) * 0.5
#             plt.arrow(i, j, modulated_x_dot[0], modulated_x_dot[1],
#                 head_width=0.008, head_length=0.01)

#     for gamma in gammas:
#         gamma.draw()
#     # plt.axis([0, 1, 0, 1])
#     plt.axis([-0.8, 0.8, 0.55, 1.1])
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.plot([x_target[0]], [x_target[1]], 'r*')
#     # plt.savefig('../images/vector_field_HBS.png', bbox_inches='tight')
#     plt.show()


# def test_HBS_svm_env():
#     '''demo of the HBS approach with SVM environment'''

#     x_target = np.array([0.9, 0.8])
#     # gamma = svm_2d_env.Environment()
#     gamma = svm_2d_env.SVMEnvironment()
#     plt.figure()
#     for i in np.linspace(-1, 1, 40):
#         for j in np.linspace(-1, 1, 40):
#             x = np.array([i, j])
#             if gamma(x) < 1:
#                 continue
#             orig_ds = linear_controller(x, x_target)
#             modulated_x_dot = modulation_HBS(x, orig_ds, [gamma]) # * 0.15
#             print("modulated x-dot Multi:", modulated_x_dot)

#             M = modulation_single_HBS(x, gamma)
#             modulated_x_dot = np.matmul(M, orig_ds.reshape(-1, 1)).flatten()
#             print("modulated x-dot Single:", modulated_x_dot)

#             modulated_x_dot = modulated_x_dot / max(1, np.linalg.norm(modulated_x_dot)) * 0.1
#             plt.arrow(i, j, modulated_x_dot[0], modulated_x_dot[1],
#                 head_width=0.008, head_length=0.01)
#     gamma.draw()
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.plot([x_target[0]], [x_target[1]], 'r*')
#     # plt.savefig('../images/vector_field_HBS.png', bbox_inches='tight')
#     plt.show()


# # --- STATUS: WORKING! --- #
# def test_HBS_learned_obs():
#     # -- Load Data and Compute Gamma Functions and Normals for obstacles --- #
#     gamma_type  = 1 # 0: single Gamma for all obstacles, 1: multiple independent Gammas
#     grid_size   = 50

#     # Using data from epfl-lasa github repo
#     X, Y = learn_gamma_fn.read_data_lasa("dynamical_system_modulation_svm/data/twoObstacles_environment.txt")
#     gamma_svm = 20
#     c_svm     = 20
#     grid_limits = [0, 1]

#     if not gamma_type:
#         # Same SVM for all Gammas (i.e. normals will be the same)
#         learned_obstacles = learn_gamma_fn.create_obstacles_from_data(data=X, label=Y, plot_raw_data=True,
#             gamma_svm=gamma_svm, c_svm=c_svm)
#     else:
#         # Independent SVMs for each Gammas (i.e. normals will be different at each grid state)
#         learned_obstacles = learn_gamma_fn.create_obstacles_from_data_multi(data=X, label=Y, plot_raw_data=True,
#             gamma_svm=gamma_svm, c_svm=c_svm)


#     # Create Data for plotting
#     xx, yy    = np.meshgrid(np.linspace(grid_limits[0], grid_limits[1], grid_size), np.linspace(grid_limits[0], grid_limits[1], grid_size))
#     positions = np.c_[xx.ravel(), yy.ravel()].T

#     # -- Draw Normal Vector Field of Gamma Functions and Modulated DS --- #
#     if not gamma_type:
#         # This will use the single gamma function formulation (which is not entirely correct due to handling of the reference points)
#         classifier        = learned_obstacles['classifier']
#         max_dist          = learned_obstacles['max_dist']
#         reference_point   = learned_obstacles[1]['reference_point']
#         gamma_svm         = learned_obstacles['gamma_svm']
#         n_obstacles       = learned_obstacles['n_obstacles']
#         filename          = "./dynamical_system_modulation_svm/figures//svmlearnedGamma_combined.png"

#         normal_vecs = learn_gamma_fn.get_normal_direction(positions, classifier, reference_point, max_dist, gamma_svm=gamma_svm)
#         fig,ax      = learn_gamma_fn.draw_contour_map(classifier, max_dist, reference_point, gamma_value=True, normal_vecs=normal_vecs,
#             grid_limits=grid_limits, grid_size=grid_size)
#         fig.savefig(filename)

#         # NF: HERE IS WHERE WE WERE WRONG BY ASSUMING WE CAN REPRESENT THE ENTIRE ENVIRONMENT WITH A SINGLE GAMMA FUNCTION
#         # AS WE NEED TO CHOOSE A REFERENCE POINT, WHAT IS THE CORRECT WAY TO CHOOOSE IT WITHOUT BREAKING CONVERGENCE?
#         # UPDATE: I TRIED A PARTITIONING APPROACH IN THE MODULATION AND IT KIND OF WORKS! IT CONVERGERS...
#         # WE NEED TO BE ABLE TO FEED ALL REFERENCE POINTS TO THESE FUNCTIONS WE CHOOSE THE CLOSEST ONE AS IN THE MODULATION!
#         gamma_vals  = learn_gamma_fn.get_gamma(positions, classifier, max_dist, reference_point)
#         normal_vecs = learn_gamma_fn.get_normal_direction(positions, classifier, reference_point, max_dist)

#         # This will use the single gamma function formulation (which
#         x_target    = np.array([0.8, 0.2])
#         # x_target    = np.array([0.65, 0.65])
#         gamma_vals  = gamma_vals.reshape(xx.shape)
#         normal_vecs = normal_vecs.reshape(2, xx.shape[0], xx.shape[1])
#         obstacle_reference_points = []
#         try:
#             for i in range(n_obstacles):
#                 obstacle_reference_points.append(learned_obstacles[i]["reference_point"])
#             print("Obstacle Reference Points", obstacle_reference_points)
#         except:
#             print ("SVM found too few obstacles (compared to ground truth)")
#         filename = "./dynamical_system_modulation_svm/figures//svmlearnedGamma_combined_modDS.png"

#         # MODULATION CONSIDERS ALL REFERENCE POINTS
#         draw_modulated_svmGamma_HBS(x_target, obstacle_reference_points, gamma_vals, normal_vecs, n_obstacles, grid_limits, grid_size, filename)

#     else:
#         for oo in range(len(learned_obstacles)):
#             classifier        = learned_obstacles[oo]['classifier']
#             max_dist          = learned_obstacles[oo]['max_dist']
#             reference_point   = learned_obstacles[oo]['reference_point']
#             gamma_svm         = learned_obstacles[oo]['gamma_svm']

#             filename          = './dynamical_system_modulation_svm/figures//svmlearnedGamma_obstacle_{}.png'.format(oo)

#             normal_vecs = learn_gamma_fn.get_normal_direction(positions, classifier, reference_point, max_dist, gamma_svm=gamma_svm)
#             fig, ax     = learn_gamma_fn.draw_contour_map(classifier, max_dist, reference_point, gamma_value=True, normal_vecs=normal_vecs, grid_limits=grid_limits, grid_size=grid_size)
#             fig.savefig(filename)

#             print("Doing modulation for obstacle {}".format(oo))
#             gamma_vals  = learn_gamma_fn.get_gamma(positions, classifier, max_dist, reference_point)

#             # This will use the single gamma function formulation (which
#             x_target    = np.array([0.8, 0.2])
#             gamma_vals  = gamma_vals.reshape(xx.shape)
#             normal_vecs = normal_vecs.reshape(2, xx.shape[0], xx.shape[1])
#             filename = "./dynamical_system_modulation_svm/figures//svmlearnedGamma_obstacle_{}_modDS.png".format(oo)
#             draw_modulated_svmGamma_HBS(x_target, reference_point, gamma_vals, normal_vecs, 1, grid_limits, grid_size, filename)

#         print("Doing combined modulation of all obstacles")
#         filename = "./dynamical_system_modulation_svm/figures//multisvmlearnedGamma_ALLobstacles_modDS.png"
#         draw_modulated_multisvmGamma_HBS(x_target, learned_obstacles, grid_limits, grid_size, filename)


# # STATUS: TESTING!
# def test_HBS_rbf_env():
#     '''demo of the HBS approach with RBF environment'''

#     # use the mouse
#     global mx, my
#     mx, my = None, None
#     def mouse_move(event):
#         global mx, my
#         mx, my = event.xdata, event.ydata

#     plt_resolution = 40

#     x_target = np.array([0.9, 0.8])
#     env = rbf_2d_env_svm.RBFEnvironment()
#     sim = rbf_2d_env_svm.PhysicsSimulator(env)

#     DRAW_INDIVIDUAL_GAMMAS = True
#     if DRAW_INDIVIDUAL_GAMMAS:
#         for gamma in env.individual_gammas:
#             plt.figure()
#             plt.axis([-1, 1, -1, 1])
#             # plt.imshow(gamma.env_img, extent=[-1.2, 1.2, -1.2, 1.2], origin='lower', cmap='coolwarm')
#             X, Y       = np.meshgrid(np.linspace(-1, 1, plt_resolution), np.linspace(-1, 1, plt_resolution))
#             V, U       = np.meshgrid(np.linspace(-1, 1, plt_resolution), np.linspace(-1, 1, plt_resolution))
#             gamma_vals = np.ndarray(X.shape)
#             for i in range(plt_resolution):
#                 for j in range(plt_resolution):
#                     x_query         = np.array([X[i, j], Y[i,j]])
#                     gamma_vals[i,j] = gamma(x_query)
#                     orig_ds         = linear_controller(x_query, x_target)
#                     modulated_x_dot = modulation_HBS(x_query, orig_ds, gammas=[gamma]) # * 0.15
#                     modulated_x_dot = modulated_x_dot / np.linalg.norm(modulated_x_dot +  epsilon) * 0.025
#                     U[i,j]     = modulated_x_dot[0]
#                     V[i,j]     = modulated_x_dot[1]

#             plt.streamplot(X,Y,U,V, density = 6, linewidth=0.55, color='k')
#             levels    = np.array([0, 1, 2])
#             cs0       = plt.contour(X, Y, gamma_vals, levels, origin='lower', colors='k', linewidths=1)
#             cs        = plt.contourf(X, Y, gamma_vals, np.arange(-3, 3, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
#             cbar      = plt.colorbar(cs)
#             cbar.add_lines(cs0)
#             plt.show()


#     x, y = -1, -1
#     sim.reset([x, y])
#     plt.figure()
#     plt.ion()
#     plt.connect('motion_notify_event', mouse_move)
#     plt.imshow(env.env_img, extent=[-1.2, 1.2, -1.2, 1.2], origin='lower', cmap='coolwarm')
#     plt.axis([-1, 1, -1, 1])
#     plt.gca().set_aspect('equal')
#     # plt.gca().get_xaxis().set_visible(False)
#     # plt.gca().get_yaxis().set_visible(False)
#     plt.show()
#     agent_plot, = plt.plot([-1], [-1], 'C2o')

#     curr_loc = sim.agent_pos()

#     reading = env.lidar(sim.agent_pos())
#     plotted = rbf_2d_env.plot_lidar(curr_loc, reading)

#     X, Y = np.meshgrid(np.linspace(-1, 1, plt_resolution), np.linspace(-1, 1, plt_resolution))
#     V, U = np.meshgrid(np.linspace(-1, 1, plt_resolution), np.linspace(-1, 1, plt_resolution))
#     gamma_vals = np.ndarray(X.shape)
#     for i in range(plt_resolution):
#         for j in range(plt_resolution):
#             x_query = np.array([X[i, j], Y[i,j]])
#             gamma_vals[i,j] = min([gamma(x_query) for gamma in env.individual_gammas])
#             orig_ds = linear_controller(x_query, x_target)
#             modulated_x_dot = modulation_HBS(x_query, orig_ds, gammas=env.individual_gammas) # * 0.15
#             modulated_x_dot = modulated_x_dot / np.linalg.norm(modulated_x_dot + epsilon) * 0.025
#             # plt.arrow(X[i, j], Y[i, j], modulated_x_dot[0], modulated_x_dot[1],
#             #     head_width=0.008, head_length=0.01)
#             U[i,j]     = modulated_x_dot[0]
#             V[i,j]     = modulated_x_dot[1]

#     plt.streamplot(X,Y,U,V, density = 6, linewidth=0.55, color='k')
#     levels    = np.array([0, 1])
#     cs0       = plt.contour(X, Y, gamma_vals, levels, origin='lower', colors='k', linewidths=1)
#     cs        = plt.contourf(X, Y, gamma_vals, np.arange(-3, 3, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
#     cbar      = plt.colorbar(cs)
#     cbar.add_lines(cs0)
#     plt.show()

#     while True:
#         if mx is not None and my is not None:
#             curr_loc = sim.agent_pos()
#             orig_ds = linear_controller(curr_loc, x_target)
#             d = modulation_HBS(curr_loc, orig_ds, gammas=env.individual_gammas)
#             d = d / np.linalg.norm(d) * 0.03
#             sim.step(d)
#             min_gamma =  min([gamma(x_query) for gamma in env.individual_gammas])
#             print("min_gamma:", min_gamma)
#             print("ds:", d)
#             x, y = sim.agent_pos()
#             agent_plot.set_data([x], [y])
#             reading = env.lidar([x, y])
#             plotted = rbf_2d_env.plot_lidar([x, y], reading, plotted)
#         for indv_gamma in env.individual_gammas:
#             pt_x, pt_y = indv_gamma.center
#             plt.plot(pt_x, pt_y, 'go', markersize=10)
#             # plt.plot(indv_gamma.boundary_points.T[0,:],indv_gamma.boundary_points.T[1,:], 'yo', markersize=2)

#         plt.plot(0.9,0.8, 'y*', markersize=60)
#         plt.pause(0.1)


# if __name__ == '__main__':

    # Test to compare modulation implementations -- working!
    # test_HBS_learned_obs() # Using environment/obstacles learned as svm-defined gamma functions

    # Test to evaluate environment and gamma function definitions -- work in progress!
    # test_HBS_rbf_env()     # Using environment defined as sum of rbf functions gamma functions
    # test_HBS_fixed_obs()
    # test_HBS_svm_env()


    # --- Test to evaluate environment and gamma function definitions -- work in progress! --- #
    # test_HBS_rbf_env()     # Using environment defined as sum of rbf functions gamma functions
    # test_HBS_fixed_obs_table() # Using environment/obstacles defined as parametrized gamma functions
