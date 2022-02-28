import numpy as np
from scipy.spatial.distance import directed_hausdorff


def hausdorff_score(prediction, groundtruth):
    hd = [directed_hausdorff(prediction[i], groundtruth[i])[0] for i in range(np.shape(groundtruth)[0])]
    return np.mean(hd)
