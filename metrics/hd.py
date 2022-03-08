import SimpleITK as sitk
import numpy as np
import scipy.spatial
import seg_metrics.seg_metrics as sg
from medpy.metric import hd95, hd
from scipy.spatial.distance import directed_hausdorff


def hausdorff_score(prediction, groundtruth):
    hd_d = [directed_hausdorff(prediction[i], groundtruth[i])[0] for i in range(np.shape(groundtruth)[0])]
    return np.mean(hd_d)


def medpy_hd95(prediction, groundtruth):
    hd_95 = hd95(prediction, groundtruth, connectivity=0)
    return hd_95


def medpy_hd(prediction, groundtruth):
    hd_d = hd(prediction, groundtruth, connectivity=0)
    return hd_d


def seg_metrics_hd95(pred_path, gt_path):
    labels = [0, 1]
    metrics = sg.write_metrics(labels=labels[1:],  # exclude background
                               gdth_path=gt_path,
                               pred_path=pred_path,
                               metrics=['hd95'])
    hd95_d = metrics[0]['hd95'][0]
    return hd95_d


def sitk_hd95(pred_path, gt_path):
    """Compute the Hausdorff distance."""
    testImage, resultImage = get_images(gt_path, pred_path)
    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float('nan')

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    e_test_image = sitk.BinaryErode(testImage, (1, 1, 0))
    e_result_image = sitk.BinaryErode(resultImage, (1, 1, 0))

    h_test_image = sitk.Subtract(testImage, e_test_image)
    h_result_image = sitk.Subtract(resultImage, e_result_image)

    h_test_array = sitk.GetArrayFromImage(h_test_image)
    h_result_array = sitk.GetArrayFromImage(h_result_image)

    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    test_coordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in
                        np.transpose(np.flipud(np.nonzero(h_test_array)))]
    result_coordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in
                          np.transpose(np.flipud(np.nonzero(h_result_array)))]

    # Use a kd-tree for fast spatial search
    def get_distances_from_ato_b(a, b):
        kd_tree = scipy.spatial.KDTree(a, leafsize=100)
        return kd_tree.query(b, k=1, eps=0, p=2)[0]

    # Compute distances from test to result; and result to test
    d_test_to_result = get_distances_from_ato_b(test_coordinates, result_coordinates)
    d_result_to_test = get_distances_from_ato_b(result_coordinates, test_coordinates)
    return max(np.percentile(d_test_to_result, 70), np.percentile(d_result_to_test, 70))


def get_images(test_filename, result_filename):
    """Return the test and result images, thresholded and non-WMH masked."""
    test_image = sitk.ReadImage(test_filename)
    result_image = sitk.ReadImage(result_filename)

    # Check for equality
    assert test_image.GetSize() == result_image.GetSize()

    # Get meta data from the test-image, needed for some sitk methods that check this
    result_image.CopyInformation(test_image)

    # Remove non-WMH from the test and result images, since we don't evaluate on that
    masked_test_image = sitk.BinaryThreshold(test_image, 0.5, 1.5, 1, 0)  # WMH == 1
    non_wmh_image = sitk.BinaryThreshold(test_image, 1.5, 2.5, 0, 1)  # non-WMH == 2
    masked_result_image = sitk.Mask(result_image, non_wmh_image)

    # Convert to binary mask
    if 'integer' in masked_result_image.GetPixelIDTypeAsString():
        b_result_image = sitk.BinaryThreshold(masked_result_image, 1, 1000, 1, 0)
    else:
        b_result_image = sitk.BinaryThreshold(masked_result_image, 0.5, 1000, 1, 0)

    return masked_test_image, b_result_image
