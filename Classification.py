from sklearn.svm import SVC
import pandas as pd
import GLCM

class Classification:
    def __init__(self):
        self.glcm = GLCM.GLCM()
        dataset = pd.read_csv("E://Coding//Skripsi//raw_features.csv", index_col=0)
        self.x = dataset[dataset.columns.drop(['name', 'label'])]
        self.y = dataset['label']
        
    def self_glcm(self, img):
        properties = ['contrast', 'energy', 'correlation', 'dissimilarity', 'homogeneity', 'ASM']
        feature = self.glcm.calc_glcm(img, img_name=None, label=None, props=properties)
        return feature
    
    def label(self, feature, bestfeature):
        model = SVC(kernel='linear', C=1.0)
        model.fit(self.x.iloc[:, bestfeature], self.y)
        label = model.predict(feature.iloc[:, bestfeature])
        return label