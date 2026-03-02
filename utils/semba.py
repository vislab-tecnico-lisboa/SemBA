#############################################################################################################################
# Sematic Information Model Functions
#############################################################################################################################

import numpy as np

# Kaplan's Update Rule 
def kaplan(belief, scores):

    return np.multiply(belief,1+scores/sum(np.multiply(scores,belief))/(1+min(scores)/sum(np.multiply(scores,belief))))
    
 
# Fusion Model (Kaplan's Rule)
def fusion_model(state, scores):

    state = np.ndarray.copy(state)

    return kaplan(state, scores)


# Foveal Observation Model
def fov_observation_model(data,total_classes):

    if len(data) != 0: dat = data[:,4:(total_classes+4)]

    return dat