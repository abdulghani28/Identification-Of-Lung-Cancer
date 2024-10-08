from skimage.feature import greycomatrix, greycoprops
import numpy as np

class GLCM:        
    def calc_glcm(self, img, img_name, label, props, dists=[5], agls=[0], lvl=256, sym=True, norm=True):
        glcm = greycomatrix(img,
                            distances=dists,
                            angles=agls,
                            levels=lvl,
                            symmetric=sym,
                            normed=norm)
        
        feature = []
        glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
        if label is not None:
            feature.append(img_name)
            feature.append(label)
        for item in glcm_props:
            feature.append(item)
    
        return feature
    
    def glcm(self, images, labels, names):
        properties = ['contrast', 'energy', 'correlation', 'dissimilarity', 'homogeneity', 'ASM']
        glcm_all = []
        for img, name, label in zip(images, names, labels):
            glcm_all.append(
                self.calc_glcm(img, name, label, props=properties)
            )
            
        columns = []
        columns.append("name")
        columns.append("label")
        for name in properties :
            columns.append(name)
        return glcm_all, columns
    