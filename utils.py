color_map = {
    'across':np.array([255,76,178])/255, # pink
    'within1':np.array([0,87,154])/255, # light blue - right hemisphere
    'within2':np.array([111,192,255])/255, # dark blue - left hemisphere
    'within':np.array([0,144,255])/255, # medium blue - collapsed across both hemispheres
    'independent':np.array([200,200,200])/255 # gray
}

def jitter(length=1):
    spacing = 0.2
    return np.random.uniform(low=-spacing,high=spacing,size=length)