import numpy as np

def estimate_affine_matrix(x,X):
    '''
    Using golden standard algorithm to estimate
    the affine transformation from 3D coordinate
    to 2D coord.
    Parameters:
    X: (3,n) matrix, where n represents number of points
    x: (2,n) matrix
    Return P that satisfy:
    x=PX
    '''
    # Normalizing x
    mean=np.mean(x,axis=1,keepdims=True)
    x=x-mean
    avg_norm=np.mean(np.sqrt(np.sum(x**2,axis=0)))
    scale=np.sqrt(2)/avg_norm
    x=scale*x # Divide by standard deviation

    # Constructing T matrix
    T=np.zeros((3,3),dtype=np.float32)
    T[0,0]=scale;T[1,1]=scale
    T[0,2]=-scale*mean[0,0]
    T[1,2]=-scale*mean[1,0]
    T[2,2]=1

    # Normalizing X
    mean=np.mean(X,axis=1,keepdims=True)
    X=X-mean
    avg_norm=np.mean(np.sqrt(np.sum(X**2,axis=0)))
    scale=np.sqrt(3)/avg_norm
    X=scale*X

    # Constructing U matrix
    U=np.zeros((4,4),dtype=np.float32)
    U[0,0]=scale;U[1,1]=scale;U[2,2]=scale
    U[0,3]=-scale*mean[0,0]
    U[1,3]=-scale*mean[1,0]
    U[2,3]=-scale*mean[2,0]
    U[3,3]=1

    num_points=x.shape[1] # number of points
    # Compute P_norm
    A=np.zeros((2*num_points,8),dtype=np.float32)
    A[0:2*num_points:2,0:3]=X.transpose()
    A[0:2*num_points:2,3]=1
    A[1:2*num_points:2,4:7]=X.transpose()
    A[1:2*num_points:2,7]=1
    b=np.reshape(x.T,[2*num_points,1])

    coef,_,_,_=np.linalg.lstsq(A,b)
    P_norm=np.zeros((3,4),dtype=np.float32)
    P_norm[0,0]=coef[0];P_norm[0,1]=coef[1];P_norm[0,2]=coef[2];P_norm[0,3]=coef[3]
    P_norm[1,0]=coef[4];P_norm[1,1]=coef[5];P_norm[1,2]=coef[6];P_norm[1,3]=coef[7]
    P_norm[2,3]=1
    return np.linalg.inv(T).dot(np.dot(P_norm,U))

def P2rst(P):
    '''
    Get the translation, rotation matrix and scaling
    factor from matrix P
    Parameters:
    P: an (3,4) matrix
    Return:
    s: scaling factor
    R: rotation matrix
    t: translation
    '''
    t=P[:2,3]
    r1=P[0,:3];r2=P[1,:3]
    scale=(np.linalg.norm(r1)+np.linalg.norm(r2))/2
    
    # Recalculate matrix R
    r1=r1/np.linalg.norm(r1)
    r2=r2/np.linalg.norm(r2)
    r3=np.cross(r1,r2)

    # Forming R
    R=np.zeros((3,3),dtype=np.float32)
    R[0]=r1
    R[1]=r2
    R[2]=r3

    return t,R,scale

def estimate_rotation_matrix(angles):
    '''
    Estimating rotation matrix from angles.
    Parameters:
    angles: a list represent the pitch, yaw, roll angle
    '''
    Rx=np.array([[1.,     0.,               0.],
                [0.,np.cos(angles[0]),np.sin(angles[0])],
                [0,-np.sin(angles[0]),np.cos(angles[0])]])
    
    Ry=np.array([[np.cos(angles[1]),0,-np.sin(angles[1])],
                [0.,                1.,         0.],
                [np.sin(angles[1]),0,np.cos(angles[1])]])
    
    Rz=np.array([[np.cos(angles[2]),np.sin(angles[2]),0.],
                [-np.sin(angles[2]),np.cos(angles[2]),0.],
                [0.,                    0.,           1.]])
    R=np.dot(Rz,np.dot(Ry,Rx))
    return R

def transform_vertices(vertices,s,R,t):
    '''
    Transform vertices, using scaling factor, rotation matrix, and translation vector
    This can be interpreted as:
    new_ver=sRvertices+t
    Parameters:
    vertices: list of vertices, in the form of [3,n_vertices]
    s: scaling factor, a scalar
    R: rotation matrix, [3,3] matrix
    t: translation vector, can be a list
    '''
    t3d=np.reshape(np.array([t]),[3,1]) # convert to numpy array
    new_vertices=s*np.dot(R,vertices)+t3d
    return new_vertices

def to_image_coord(vertices,w,h):
    '''
    Transform from camera coordinates to image coordinates
    Parameters
    vertices: list of vertices, of shape [n_vertices,3]
    w: width of the image
    h: height of the image
    '''
    image_vertices=vertices.copy()
    # Since camera coordinate's origin is at the center of image, we need to add an extra to point coordinate in image plane
    image_vertices[:,0]=image_vertices[:,0]+w/2 # x axis
    image_vertices[:,1]=image_vertices[:,1]+h/2 # y axis
    # flip the y axis
    image_vertices[:,1]=h-1-image_vertices[:,1]
    return image_vertices

if __name__=='__main__':
    x=np.random.randn(2,10)
    X=np.random.randn(3,10)
    P_affine=estimate_affine_matrix(x,X)
    print(P_affine)
    t,R,scale=P2rst(P_affine)
    print(t)
    print(R)
    print(scale)