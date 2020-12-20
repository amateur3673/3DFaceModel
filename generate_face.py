import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
from imutils import face_utils
import imutils
from morphable import MorphableModel
from process import estimate_rotation_matrix,transform_vertices,to_image_coord
from face3d import mesh
from fitting import fitting_landmarks

def generate_random_face():
    '''
    Generate a random face from Basel Face Model
    '''
    bfm=MorphableModel(model_path='BFM/BFM.mat') # basel face model
    sp=np.random.rand(bfm.n_shape_para,1)*1e04 # shape coef
    ep=np.zeros((bfm.n_exp_para,1))
    vertices=bfm.model['shapeMU']+np.dot(bfm.model['shapePC'],sp)+np.dot(bfm.model['expPC'],ep)
    vertices=np.reshape(vertices,[-1,3]).T #reshape to the shape [n_vertices,3]
    tp=np.random.rand(bfm.n_tex_para,1) # texture coef
    colors=bfm.model['texMU']+np.dot(bfm.model['texPC'],tp*bfm.model['texEV'])
    colors=np.reshape(colors,[-1,3])/255.0
    s=8e-4
    t=[0,0,0]
    angles=[10*np.pi/180,20*np.pi/180,3*np.pi/180]
    R=estimate_rotation_matrix(angles) # Rotation matrix
    print(R)
    transformed_vertices=transform_vertices(vertices,s,R,t) # transformed vertices of shape [3,n_vertices]
    image_vertices=to_image_coord(transformed_vertices.T,w=256,h=256)
    #image_vertices=mesh.transform.to_image(transformed_vertices.T,h=256,w=256)
    img=mesh.render.render_colors(image_vertices,bfm.model['tri'],colors,h=256,w=256,c=3)
    img=np.clip(img,0,1)
    plt.imshow(img)
    plt.show()
    kpts=bfm.model['kpt_ind']
    lm=transformed_vertices[:2,kpts].T
    s,R,t,sp,ep=fitting_landmarks(np.reshape(lm,[-1,1]),kpts,bfm,n_iters=3)
    print(s,t,R)
    vertices=bfm.model['shapeMU']+np.dot(bfm.model['shapePC'],sp)+np.dot(bfm.model['expPC'],ep)
    vertices=np.reshape(vertices,[-1,3]).T
    t=[t[0],t[1],1]
    transformed_vertices=transform_vertices(vertices,s,R,t)
    image_vertices=to_image_coord(transformed_vertices.T,w=256,h=256)
    img=mesh.render.render_colors(image_vertices,bfm.model['tri'],colors,h=256,w=256)
    plt.imshow(img)
    plt.show()

def generate_facial_img(bfm,sp,ep,tp,s,R,t,h=256,w=256):
    '''
    Generate face from Basel Face Model
    Parameters:
    sp: coefficient of identity (shape), which is a vector of 99 dimension
    ep: coefficient of expression (exp), which is a vector of 29 dimension
    tp: coefficient of texture (tex), which is a vector of 99 dimension
    s: scaling factor, a scalar
    R: rotation matrix, of shape [3,3]
    t: translation vector of x,y,z coordinate, can be a list of 3 elements
    h: height of render image
    w: width of render image
    '''
    print('Generate facial image ...')
    print('Init Basel Face Model ...')
    vertices=bfm.model['shapeMU']+np.dot(bfm.model['shapePC'],sp)+np.dot(bfm.model['expPC'],ep)
    vertices=np.reshape(vertices,[-1,3]).T #reshape to the shape [n_vertices,3]
    colors=bfm.model['texMU']+np.dot(bfm.model['texPC'],tp)
    colors=np.reshape(colors,[-1,3])/255.0 # Rescaling to [0,1] scale
    # Apply rotation and translation to the vertice coordinate
    transformed_vertices=transform_vertices(vertices,s,R,t)
    # Convert to image coordinate
    image_vertices=to_image_coord(transformed_vertices.T,w,h)
    # Render to image
    img=mesh.render.render_colors(image_vertices,bfm.model['tri'],colors,h,w)
    img=np.clip(img,0,1)
    return img

def get_facial_landmarks(img,lm_path='BFM/shape_predictor_68_face_landmarks.dat'):
    '''
    Get 68 facial landmarks
    Parameters:
    img: numpy array represent the image
    lm_path: landmark path
    '''
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor(lm_path)
    rects=detector(gray,1)

    for i, rect in enumerate(rects):
        shape=predictor(gray,rect)
        shape=face_utils.shape_to_np(shape)

    return shape

def estimate_head_pose(img_path,h=256,w=256):
    bfm=MorphableModel(model_path='BFM/BFM.mat')
    tp=np.random.rand(bfm.n_tex_para,1)*bfm.model['texEV']
    img=cv2.imread(img_path)
    img=cv2.resize(img,(w,h))
    lm=get_facial_landmarks(img)
    lm[:,0]=lm[:,0]-w/2
    lm[:,1]=h-1-lm[:,1]
    lm[:,1]=lm[:,1]-h/2
    lm=np.reshape(lm,[-1,1])
    s,R,t,sp,ep=fitting_landmarks(lm,bfm.model['kpt_ind'],bfm)
    t=[t[0],t[1],1]
    render_img=generate_facial_img(bfm,sp,ep,tp,s,R,t)
    print('s R t are: {}{}{}'.format(s,R,t))
    return render_img

if __name__=='__main__':
    
    bfm=MorphableModel(model_path='BFM/BFM.mat')
    '''
    sp=np.random.rand(bfm.n_shape_para,1)*1e04
    ep=np.zeros((bfm.n_exp_para,1))
    tp=np.random.rand(bfm.n_tex_para,1)*bfm.model['texEV']
    s=8e-4
    t=[0,0,0]
    R=estimate_rotation_matrix(angles=[0,0,0])
    img=generate_facial_img(bfm,sp,ep,tp,s,R,t,h=256,w=256)
    plt.imshow(img)
    plt.show()
    '''
    render_img=estimate_head_pose(img_path='choupham.jpg')