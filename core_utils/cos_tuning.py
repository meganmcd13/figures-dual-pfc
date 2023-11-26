import numpy as np

class cos_tuning:
    
    def __init__(self):
        return


    def fit(self,angs,x):
        if not(x.flatten().shape[0]==x.shape[0]):
            raise TypeError('"x" must be a flat numpy vector')
        if not(angs.flatten().shape[0]==angs.shape[0]):
            raise TypeError('"angs" must be a flat numpy vector')
            
        self.params = dict()
        angs_rad = np.deg2rad(angs)
        A = np.column_stack((np.ones(len(angs_rad)), np.cos(angs_rad), np.sin(angs_rad)));
        betas = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(x.T);
        
        bl = betas[0]
        PD = np.arctan2(betas[2],betas[1])
        if PD<0:
            PD = PD + 2*np.pi
        mod_val = betas[1] / np.cos(PD)
        mod_depth = mod_val/bl
        self.params = {
                'bl':bl,
                'PD':np.rad2deg(PD),
                'mod_val':mod_val,
                'mod_depth':mod_depth
            }
        return self.params['mod_depth'],self.params['PD'],self.params['mod_val'],self.params['bl']

    def get_params(self):
        return self.params

    def predict(self,angs):
        angs_rad = np.deg2rad(angs)
        PD_rad = np.deg2rad(self.params['PD'])
        mod_val = self.params['mod_val']
        bl = self.params['bl']
        return bl + mod_val * np.cos(angs_rad-PD_rad)
    