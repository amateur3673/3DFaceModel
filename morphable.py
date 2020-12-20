import numpy as np
from scipy.io import loadmat
from process import estimate_affine_matrix,P2rst

class MorphableModel:
    def __init__(self,model_path):
        '''
        Model info
        shapeMU: [3*nver,1]
        tri: [ntri,3]
        shapePC: [3*nver,199]
        expPC: [3*nver,29]
        texPC: [3*nver,199]
        shapeEV:[199,1]
        expPC: [29,1]
        texEV: [199,1]
        '''
        self.model=self.preprocess(model_path)
        self.nver=self.model['shapeMU'].shape[0]//3 # number of vertice
        self.ntri=self.model['tri'].shape[0] # number of triangles
        self.n_shape_para=self.model['shapePC'].shape[1] # number of identity parameters
        self.n_exp_para=self.model['expPC'].shape[1] # number of expression parameters
        self.n_tex_para=self.model['texPC'].shape[1] # number of texture parameters

        self.kpt_ind=self.model['kpt_ind'] # 68 landmark index
        self.triangles=self.model['tri'] # triangle for forming image in 3DMM
        self.full_triangles=np.vstack((self.model['tri'],self.model['tri_mouth']))
        
    def preprocess(self,model_path):
        '''
        Loading the 3DFFA model
        Preprocess idea from face3d repo
        '''
        data=loadmat(model_path)
        model=data['model']
        model=model[0,0]
        model['shapeMU']=(model['shapeMU']+model['expMU']).astype(np.float32)
        model['shapePC']=model['shapePC'].astype(np.float32)
        model['shapeEV']=model['shapeEV'].astype(np.float32)
        model['expPC']=model['expPC'].astype(np.float32)
        model['expEV']=model['expEV'].astype(np.float32)

        model['tri']=model['tri'].T.copy(order='C').astype(np.int32)-1
        model['tri_mouth']=model['tri_mouth'].T.copy(order='C').astype(np.int32)-1

        model['kpt_ind']=(np.squeeze(model['kpt_ind'])-1).astype(np.int32) 

        return model       