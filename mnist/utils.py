#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time




#get average weights
def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] *= marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = torch.div(w_avg[key], sum(marks))
    return w_avg

#get accumulated gradients
def accumulated_updates(model, old_delta_weights, local_weights, lr, momentum):
    accumulated_updates = []
    update = copy.deepcopy(local_weights[0])
    for key in update.keys():
        update[key] *= 0
    for i in range(len(local_weights)):
        accumulated_updates.append(copy.deepcopy(update))

    for i in range(len(local_weights)):
        for key in update.keys():
            model[key] = torch.squeeze(model[key])
            old_delta_weights[i][key] = torch.squeeze(old_delta_weights[i][key])
            local_weights[i][key] = torch.squeeze(local_weights[i][key])
    
    for i in range(len(local_weights)):
        for key in update.keys():
            delta_key = local_weights[i][key] - model[key]
            key_update = (momentum*old_delta_weights[i][key] - delta_key)/lr
            accumulated_updates[i][key] = key_update
    return accumulated_updates

#Compute distance between two n-dimensional arrays

def get_l2_norm_distance(coords1, coords2):

    d = coords1.data.cpu().numpy() - coords2.data.cpu().numpy()
    d = np.power(d,2)
    d = np.sum(d)
    d = np.sqrt(d)
    return np.max(d, 0)


def get_centroid(coords):

    minimum_distance = np.power(10,30)
    centroid = coords[0]
    for i in range(0, len(coords)):
        temp_centroid = coords[i]
        coord_summed_distances = 0
        for j in range(0, len(coords)):
            coord_summed_distances +=  torch.dist(temp_centroid.squeeze().cpu().float(), coords[j].squeeze().cpu().float())

        if(coord_summed_distances < minimum_distance):
            minimum_distance = coord_summed_distances
            centroid = temp_centroid
        elif(coord_summed_distances == minimum_distance):
            centroid = (centroid + temp_centroid)/2
            
    return centroid, minimum_distance



def get_current_round_centroid(updates):

    start_time = time.time()
    layers_weights = copy.deepcopy(updates[0])
    centroid = copy.deepcopy(updates[0])
    distances = copy.deepcopy(updates[0])
    last_layer_updates = []
    
    for key in layers_weights.keys():
        layers_weights[key] = []
        centroid[key] = []
        distances[key] = []

    for key in layers_weights.keys():
        for i in range(len(updates)):
            layers_weights[key].append(updates[i][key])
    
    stop_time = time.time()
    
    for key in layers_weights.keys():
        centroid[key], distances[key] = get_centroid(layers_weights[key])
    return centroid, distances, stop_time-start_time
    



def get_distances_to_centroid(updates, centroid, normalized = False):
    
    temp_dic = centroid
    total_distances = []


    layers_elements = {}
    total = 0
    for key in temp_dic.keys():
        layers_elements[key] = temp_dic[key].numel()
        total+=temp_dic[key].numel()
    layers_elements['total'] = total
    

    for i in range(len(updates)):
        d = 0
        for key in centroid.keys():
            if(normalized):
                d+= get_l2_norm_distance(centroid[key], updates[i][key])*(layers_elements[key]/layers_elements['total'])
            else:
                d+= get_l2_norm_distance(centroid[key], updates[i][key])
        total_distances.append(d)

    return  total_distances


def get_layer_weights(updates, layer_name):
    
    layer_weights = []
    for i in range(len(updates)):
        layer_weights.append(updates[i][layer_name].data.cpu().numpy())

    return layer_weights
        
def compute_mad(update, update_length):
    m = np.median(update)
    mad = 0
    for i in range(update_length):
        mad+=abs(update[i]-m)
    
    return mad/update_length




def plot_box_whishker(data):

    ax = sns.boxplot(y = data)
    outliers = [y for stat in boxplot_stats(data) for y in stat['fliers']]
    print('\n\nOutliers\n')
    print(outliers)
    for y in outliers:
        ax.plot(1, y, 'p')
    ax.set_xlim(right=1.5)
    plt.show()



# get median value, q1 and q3 indices
def get_median_and_quartiles_indices(distances):
    
    l = len(distances)
    if(l%2==0):
        i1 = int(l/2) - 1
        i2 = int(l/2)
        return (sum(distances[i1:i2+1])/2), i1, i2
    else:
        i1 = int(l/2)
        return distances[i1], i1,i1




# get quartiles (q1, median/q2, q3 and iqr)
def get_quartiles(distances):

    q1, q3= np.percentile(distances,[25,75], interpolation='nearest')
    iqr = q3 - q1
    
            
    return q1,  q3, iqr


    


#generate a random lables out of the labels range to launch an untargeted attack
def get_random_untargeted_attack_list(num_classes, target_length):
    
        return torch.randint(0, num_classes, (target_length,))

#generate a random lables inside the labels range to launch an targeted attack
def get_random_targeted_attack_list(real_target, num_classes):
    
    added_indices = []
    fake_target = real_target.clone().detach()
    for t1 in real_target:
        a = np.random.randint(num_classes)
        for indx, t2 in enumerate(fake_target):
            while(a==t1):
                a = np.random.randint(num_classes)
            if(t2 == t1 and indx not in added_indices):
                fake_target[indx] = a
                added_indices.append(indx)
                
            
    return fake_target

# set known real labels to known  fake labels
def set_targeted_attack_list(real_target, fake_target):
    return fake_target

    #Krum algorithm 

def distance(update1, update2):

    update1_list = list(update1.parameters())
    update2_list = list(update2.parameters())
    
    dist = 0

    for i in range(len(update1_list)):
        dist+= torch.dist(update1_list[i], update2_list[i], 2)
    
    del update1_list
    del update2_list
    
    return dist


def krum(update_list, f, multi = True):
    score_list = []
    for i in range(len(update_list)):
        dist_list = []
        for j in range(len(update_list)):
            dist_list.append(distance(update_list[i],update_list[j]))
        dist_list.sort()
        truncated_dist_list = dist_list[:-(f+1)]
        score = sum(truncated_dist_list)
        score_list.append(score)
    sorted_score_indices = np.argsort(np.array(score_list))

    if multi:
    	return sorted_score_indices[:-f]	
    else:
    	return sorted_score_indices[0]





#Geometric median algorithms

def geometric_median(points, method='auto', options={}):
    """
    Calculates the geometric median of an array of points.
    method specifies which algorithm to use:
        * 'auto' -- uses a heuristic to pick an algorithm
        * 'minimize' -- scipy.optimize the sum of distances
        * 'weiszfeld' -- Weiszfeld's algorithm
    """

    points = np.asarray(points)

    if len(points.shape) == 1:
        # geometric_median((0, 0)) has too much potential for error.
        # Did the user intend a single 2D point or two scalars?
        # Use np.median if you meant the latter.
        raise ValueError("Expected 2D array")

    if method == 'auto':
        if points.shape[1] > 2:
            # weiszfeld tends to converge faster in higher dimensions
            method = 'weiszfeld'
        else:
            method = 'minimize'

    return _methods[method](points, options)


def minimize_method(points, options={}):
    """
    Geometric median as a convex optimization problem.
    """

    # objective function
    def aggregate_distance(x):
        return cdist([x], points).sum()

    # initial guess: centroid
    centroid = points.mean(axis=0)

    optimize_result = minimize(aggregate_distance, centroid, method='COBYLA')

    return optimize_result.x


def weiszfeld_method(points, options={}):
    """
    Weiszfeld's algorithm as described on Wikipedia.
    """

    default_options = {'maxiter': 1000, 'tol': 1e-7}
    default_options.update(options)
    options = default_options

    def distance_func(x):
        return cdist([x], points)

    # initial guess: centroid
    guess = points.mean(axis=0)

    iters = 0

    while iters < options['maxiter']:
        distances = distance_func(guess).T

        # catch divide by zero
        # TODO: Wikipedia cites how to deal with distance 0
        distances = np.where(distances == 0, 1, distances)

        guess_next = (points/distances).sum(axis=0) / (1./distances).sum(axis=0)

        guess_movement = np.sqrt(((guess - guess_next)**2).sum())

        guess = guess_next

        if guess_movement <= options['tol']:
            break

        iters += 1

    return guess


_methods = {
    'minimize': minimize_method,
    'weiszfeld': weiszfeld_method,
}



def get_distances_from_geomed(geomed, points):
    return cdist([geomed], points)


def get_honest_workers(distances_from_geomed):
   
    distances_from_geomed = np.asarray(distances_from_geomed)
    indices = distances_from_geomed.argsort()
    sorted_distances = np.asarray(sorted(distances_from_geomed))

    n = len(sorted_distances)
    between_distances = np.zeros(n)
    for i in range(1, n):
        between_distances[i] = sorted_distances[i]-sorted_distances[i-1]
    
    m = int(n/2)+1    
    tolerance = max(between_distances[:m+1])*1.5
    scores = np.ones(n)
    for i in range(m, n):
        if between_distances[i] > tolerance:
            scores[i:] = 0
            break
    sorted_scores = np.empty_like(scores)
    
    for i, idx in enumerate(indices):
        sorted_scores[idx] = scores[i]
   
    scores = sorted_scores
    median = np.percentile(distances_from_geomed, [60])
    for i in range(n):
        if distances_from_geomed[i]<=median:
            scores[i] = 1
            
    honest_workers = [i for i, x in enumerate(scores) if x]
    
    return honest_workers, list(scores)
