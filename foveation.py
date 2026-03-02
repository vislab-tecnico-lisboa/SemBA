##########################################################################################################
# Multi-Scale Foveal Mechanism                                                                           #
# Joao Luzio, Institute for Systems and Robotics                                                         #
# Instituto Superior Técnico, Lisbon, 2026                                                               #
##########################################################################################################

import cv2 as cv

class MS_Foveation(object):

    def __init__(self, levels=4, dim=160, scale_factor=2):

        self.levels = levels
        self.base_dim = dim
        self.scale = scale_factor
        self.layers = []

    def foveate(self, img, fixation):

        self.layers = []
        x_ctr, y_ctr = fixation

        # add black padding
        pad_size = self.base_dim*(self.scale**(self.levels-1))
        img = cv.copyMakeBorder(img.copy(), 
                                pad_size, pad_size,
                                pad_size, pad_size, 
                                cv.BORDER_CONSTANT, value=(0,0,0))

        # crop each layer
        for l in range(self.levels):

            y_start = int(y_ctr-(self.base_dim*(self.scale**l))/2)+pad_size
            y_end = int(y_ctr+(self.base_dim*(self.scale**l))/2)+pad_size 
            x_start = int(x_ctr-(self.base_dim*(self.scale**l))/2)+pad_size
            x_end = int(x_ctr+(self.base_dim*(self.scale**l))/2)+pad_size
            self.layers.append(img[y_start:y_end,x_start:x_end])

        # rescale the layers to base dimension
        for l in range(1,self.levels):

            self.layers[l] = cv.resize(self.layers[l],
                                        (self.base_dim, self.base_dim),
                                        interpolation=cv.INTER_LINEAR)
            
        return self.layers
    
    def bbox_remapping(self, bbox, lvl, center):
        
        for j in range(len(bbox)): bbox[j] = bbox[j]*(self.scale**(lvl-1))
        box = [center[j%2]-(self.scale**(lvl-1))*self.base_dim/2+bbox[j] for j in range(len(bbox))]

        return box
    
    def get_fov_topology(self):

        assert self.layers != [], "can't build toplogy without first running self.foveate(...)" 

        topology = cv.resize(self.layers[self.levels-1],
                                ((self.scale**(self.levels-1))*self.base_dim,
                                (self.scale**(self.levels-1))*self.base_dim))
        ctr = int((self.scale**(self.levels-1))*self.base_dim/2)

        for l in range(self.levels-1, -1, -1):

            start = ctr - int((self.scale**l)*self.base_dim/2)

            for y in range(self.base_dim):

                for x in range(self.base_dim):

                    x_min = start + x*(self.scale**l)
                    y_min = start + y*(self.scale**l)

                    for i in range(self.scale**l):

                        for j in range(self.scale**l):

                            topology[y_min + i, x_min + j] = self.layers[l][y,x]

        return topology