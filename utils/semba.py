###################################################################################################################
# Sematic Information Model Functions for SemBA x MS-Fovea                                                        #
###################################################################################################################

import numpy as np

# Kaplan's Update Rule 
def kaplan(belief, scores):

    return np.multiply(belief,
                       1+scores/sum(np.multiply(scores,belief))/(1+min(scores)/sum(np.multiply(scores,belief))))
    
 
# Fusion Model (Kaplan's Rule)
def fusion_model(state, scores):

    state = np.ndarray.copy(state)

    return kaplan(state, scores)


# Foveal Observation Model
def fov_observation_model(data,total_classes):

    if len(data) != 0: dat = data[:,4:(total_classes+4)]

    return dat

# Target Class Attention Map
def attention_map(map, cells, class_id):

    aux = np.array(map)
    sal_map = np.zeros(cells)

    for y in range(sal_map.shape[0]):
        for x in range(sal_map.shape[1]):
            sal_map[y,x] = aux[y,x,class_id-1]/np.sum(aux[y,x,:])