import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity, mean_squared_error

def display_img(img, cmap='gray', file_name=None):
    """Uses matplotlib to display the image.
    """
    
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)

    ax.grid(False)

    if(len(img.shape) == 3):
        # matplotlib expects RGB
        ax.imshow(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img,cmap=cmap)

    if file_name is not None:
        fig.savefig(file_name, bbox_inches='tight')
    
    return

def display_2img(img_a, img_b, cmap='gray', file_name=None):
    """Plot 2 images. Assumes GRB

    """
    fig, ax = plt.subplots(1,2, figsize=(14,8), sharey=False)

    if(len(img_a.shape) == 3):
        # matplotlib expects RGB
        ax[0].imshow(
            cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
        ax[1].imshow(
            cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    else:
        ax[0].imshow(img_a, cmap=cmap)
        ax[1].imshow(img_b, cmap=cmap)

    ax[0].grid(False)
    ax[1].grid(False)


    if file_name is not None:
        fig.savefig(file_name,bbox_inches='tight')


    None

    return

def register_flat(image_name):
    """Takes 2 flat images already selected and 
    align them"""

    img_a = cv2.imread(image_name+'_a.png')
    img_b = cv2.imread(image_name+'_b.png')
    
    img_a_aligned, img_b_aligned = align_images(img_a, img_b)

    # to avoid having black frame around diff images
    img_a_aligned[img_b_aligned[:, :, :] == 0] = 0
    
    cv2.imwrite(image_name+'_a_aligned.png', img_a_aligned)
    cv2.imwrite(image_name+'_b_aligned.png', img_b_aligned)


    return


def find_differences(
        img_a, img_b, tresh_quantile=0.95, 
        ssim = True, n_diff=15):
    """Take 2 perfectly aligned images and find the differences
    using structural similarity.

    Return img_a with rectangular contours at the difference 
    positions

    n is the maximun number of differences expected.
    
    """

    #
    # 1. blurring 
    #
    
    # kernel 2% of the image size
    kernel_size = int(img_a.shape[1]/50)

    # must be odd if median
    kernel_size += kernel_size%2-1

    img_a_blurred = cv2.GaussianBlur(
        img_a, (kernel_size, kernel_size), 1.5)
    img_b_blurred = cv2.GaussianBlur(
        img_b, (kernel_size, kernel_size), 1.5)
   
    #
    # 2. difference operation 
    #

    # img_a - img_b
    if ssim:
        score, diff_ssim = structural_similarity(
            img_a_blurred, img_b_blurred,
            multichannel=True, full=True, gaussian_weights=True)

        # the diff is the opposite of the similarity
        diff = 1.0-diff_ssim
        
    else:
        diff= cv2.absdiff(img_a_blurred, img_b_blurred)
    
    # renormalise
    diff = cv2.normalize(
        diff, None, alpha=0, beta=255, 
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    diff = diff.astype(np.uint8)

    #
    # 3. binary image
    #

    diff_gray = diff.max(axis=2)

    # threshold is set to 5% brightest pixels 
    min_thres = np.quantile(diff_gray, tresh_quantile)

    # simple thresholding to create a binary image
    ret, thres = cv2.threshold(diff_gray, min_thres, 255, cv2.THRESH_BINARY)

    # opening operation to clean the noise with a small kernel
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=3)

    # and dilatation operation to increase the size of elements 
    kernel_dilate = np.ones((5,5),np.uint8)
    diff_gray_thres = cv2.dilate(opening, kernel_dilate, iterations=2)

    #
    # 4. difference components
    #

    components = largest_components(diff_gray_thres, n_diff)

    #
    # 5. overlay found differences
    #

    img_final = img_a.copy()
      
    for component in components:
            
        x,y,w,h = component[:4]
    
        pt1 = (x,y)
        pt2 = (x+w,y+h)

        cv2.rectangle(
            img_final, pt1=pt1, pt2=pt2,
            color=(0,0,255), thickness=8)
    
    return img_final

def largest_components(
        binary_img, n, remove_borders=True):
    """Take a binary image and return the nst
    largest components.

    If the number of component is less, return 
    all components
    
    If remove_borders is set, it will remove 
    all components that at least half the 
    image width or length

    stats array    
    cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
    cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
    cv2.CC_STAT_WIDTH The horizontal size of the bounding box
    cv2.CC_STAT_HEIGHT The vertical size of the bounding box

    """

    # detect connected components
    retval, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(binary_img)

    if remove_borders:
        img_w, img_h = binary_img.shape

        components = []
        for i, stat in enumerate(stats):  
            x,y,w,h = stat[0:4]
            # remove outer border
            if (w > img_w*0.5) | (h > img_h*0.5):
                continue
            
            components.append(stat)
        components = np.array(components)
        
    else:
        components = stats

    # keep the n largest components
    # based on area
    try:
        # sort based on the 5th column (the area)
        sorted_indices = components[:,4].argsort()
        
        # keep the 15 largest elements
        largest_components = components[sorted_indices][-n:]
    except:
        pass

    return largest_components


def align_images(im1, im2, warp_mode = cv2.MOTION_TRANSLATION):
    """Taken from https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    See also, for future developments: https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

    """
    
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
        
    # Find size of image1
    sz = im1.shape

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    # Show final results
    #cv2.imshow("Image 1", im1)
    #cv2.imshow("Image 2", im2)
    #cv2.imshow("Aligned Image 2", im2_aligned)
    #cv2.waitKey(0)

    return im1, im2_aligned
    
