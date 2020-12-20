import numpy as np
import cv2
import dlib
from imutils import face_utils
import imutils
from process import estimate_affine_matrix,P2rst,estimate_rotation_matrix

def estimate_shape_params(lm,shapeMU,shapePC,exp,s,R,t,shapeEV,lambd):
    '''
    Estimating shape parameters
    Parameters:
    lm: landmark, of shape [2n,1]
    shapeMU: mean shape, of shape [3n,1]
    shapePC: identity parameters, of shape [3n,99]
    exp: expression, of shape [3n,1]
    s: scaling factor, a scalar
    R: rotation matrix, of shape [3,3]
    t: translation matrix, of shape [2,1] (translation of x and y coord)
    shapeEV: identity std, for regularization, of shape [99,1]
    lambd: regularization term
    '''
    n_vertices=shapePC.shape[0]//3
    n_feats=shapePC.shape[1]
    t=np.array(t)
    u=shapeMU+exp
    P=np.array([[1,0,0],[0,1,0]],dtype=np.float32)
    A=s*np.dot(P,R)
    b=np.dot(A,np.reshape(u,[-1,3]).T)
    b=np.reshape(b,[-1,1])+np.tile(t[:,np.newaxis],[n_vertices,1])
    U=np.zeros((3,n_vertices*n_feats))
    U[0,:]=shapePC[0:3*n_vertices+1:3,:].flatten()
    U[1,:]=shapePC[1:3*n_vertices+1:3,:].flatten()
    U[2,:]=shapePC[2:3*n_vertices+1:3,:].flatten()
    D=np.reshape(np.dot(R[:2,:],U),[2,n_vertices,n_feats])
    D=np.reshape(np.transpose(D,[1,0,2]),[-1,n_feats])
    # We have the equation
    right_eq=np.dot(D.T,lm-b)
    shapeEV=1/shapeEV.flatten()
    left_eq=np.dot(D.T,D)+lambd*np.diag(shapeEV**2)
    sp=np.linalg.inv(left_eq).dot(right_eq)
    return sp

def estimate_expression_params(lm,shapeMU,expPC,shape,s,R,t,expEV,lambd):
    '''
    Estimate expression parameters.
    Parameters:
    lm: landmark array position, of shape [2n,1]
    shapeMU: mean shape coefficient, in shape [3n,1]
    expPC: expression parameters, of shape [3n,29]
    shape: shape, of shape [3n,1]
    s: scaling factor, a scalar
    R: rotation matrix, of shape [3,3]
    t: translation matrix, of shape [3,1]
    expEV: expression std, for regularization purpose
    lambd: regularization parameters
    '''
    n_vertices=expPC.shape[0]//3
    n_feats=expPC.shape[1]
    t=np.array(t)
    u=shapeMU+shape
    P=np.array([[1,0,0],[0,1,0]],dtype=np.float32)
    A=s*np.dot(P,R)
    b=np.dot(A,np.reshape(u,[-1,3]).T)
    b=np.reshape(b,[-1,1])+np.tile(t[:,np.newaxis],[n_vertices,1])
    U=np.zeros((3,n_vertices*n_feats))
    U[0,:]=expPC[0:3*n_vertices+1:3,:].flatten()
    U[1,:]=expPC[1:3*n_vertices+1:3,:].flatten()
    U[2,:]=expPC[2:3*n_vertices+1:3,:].flatten()
    D=np.reshape(np.dot(R[:2,:],U),[2,n_vertices,n_feats])
    D=np.reshape(np.transpose(D,[1,0,2]),[-1,n_feats])
    right_eq=np.dot(D.T,lm-b)
    expEV=1/expEV.flatten()
    left_eq=np.dot(D.T,D)+lambd*np.diagflat(expEV**2)
    ep=np.linalg.inv(left_eq).dot(right_eq)
    return ep

def fitting_landmarks(lm,index,bfm,lambd1=20,lambd2=40,n_iters=3):
    '''
    Fitting landmark
    Parameters:
    lm: array of landmark, of shape [128,1]
    index: index of fitting vertices
    bfm: MorphableModel object
    lambd1: regularizaton term for expression
    lambd2: regularization term for shape
    n_iters: number of iteration
    '''
    # Initialize sp and ep
    sp=np.zeros((bfm.n_shape_para,1))
    ep=np.zeros((bfm.n_exp_para,1))
    # Get the position for each vertice (including x,y,z position)
    index=np.tile(index[:,np.newaxis],[1,3])*3
    index[:,1]+=1
    index[:,2]+=2
    index_all=index.flatten()
    
    shapeMU=bfm.model['shapeMU'][index_all,:1]
    shapePC=bfm.model['shapePC'][index_all,:bfm.n_shape_para]
    expPC=bfm.model['expPC'][index_all,:bfm.n_exp_para]

    s=4e-04
    R=estimate_rotation_matrix(angles=[0,0,0])
    t=[0,0,0]
    expEV=bfm.model['expEV']
    shapeEV=bfm.model['shapeEV']

    for i in range(n_iters):
        print('Iteration {}:'.format(i))
        X=shapeMU+np.dot(expPC,ep)+np.dot(shapePC,sp)
        P=estimate_affine_matrix(np.reshape(lm,[-1,2]).T,np.reshape(X,[-1,3]).T)
        t,R,s=P2rst(P)

        # fitting
        shape=np.dot(shapePC,sp)
        ep=estimate_expression_params(lm,shapeMU,expPC,shape,s,R,t,expEV,lambd1)
        exp=np.dot(expPC,ep)
        sp=estimate_shape_params(lm,shapeMU,shapePC,exp,s,R,t,shapeEV,lambd2)

    return s,R,t,sp,ep