# Path: src\sift.py
import cv2


class SIFT:
    
    # SIFT object
    sift = None
    
    # to initialize the SIFT object and the directory structure
    def __init__(self):
        
        # Initialize SIFT object
        self.sift = cv2.SIFT_create()


    """
    This function computes SIFT descriptors 

    Parameters:
    - img: Image that the funciton compute descriptors for it

    Returns:
    - descriptors: the descriptors of the image
    """
    def compute(self, img):

        # Convert to grayscale
        # the image form preprocces model is grey scaled so no need
        # Detect keypoints and compute descriptors
        _, descriptors = self.sift.detectAndCompute(img, None)
        # return keypoints and descriptors
        return descriptors
    
    def compute_keypoints(self, img):

        # Convert to grayscale
        # the image form preprocces model is grey scaled so no need
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        # return keypoints and descriptors
        return (keypoints, descriptors)
    
    """
    This function draws the keypoints of  SIFT on the image 

    Parameters:
    - keypoints: keypoints of the image
    - img: Image that the funciton compute descriptors for it

    Returns:
    - None
    """
    def draw_descriptors(self, keypoints, img):
        # Convert to grayscale
        # the image form preprocces model is grey scaled so no need
        # Marking the keypoint on the image using circles
        img_keypoints =cv2.drawKeypoints(img ,keypoints ,img ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # return the image with the keypoints
        return img_keypoints
        
        
if __name__ == '__main__':
    # run the sift
    sift = SIFT()
    img1 = cv2.imread('1.jpg')
    img2 = cv2.imread('2.jpg')
    ratio1 = 480 / img1.shape[1]
    ratio2 = 480 / img2.shape[1]
    img1 = cv2.resize(img1, (480, int(img1.shape[0] * ratio1)))
    img2 = cv2.resize(img2, (480, int(img2.shape[0] * ratio2)))
    img2 = cv2.resize(img2, (480, 480))
    k1, d1 = sift.compute_keypoints(img1)
    k2, d2 = sift.compute_keypoints(img2)
    print(len(d1))
    print(len(d2))
    
    img_keypoints1 = sift.draw_descriptors(k1, img1)
    img_keypoints2 = sift.draw_descriptors(k2, img2)
    # match keypoints from img1 and img2
    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(d1,d2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1, k1, img2, k2, matches[:50], img2, flags=2)
    # save the image with the keypoints on it
    cv2.imwrite('test_sift1.jpg', img_keypoints1)
    cv2.imwrite('test_sift2.jpg', img_keypoints2)
    cv2.imwrite('test_sift3.jpg', img1)
    cv2.imwrite('test_sift4.jpg', img2)
    cv2.imwrite('test_sift5.jpg', img3)
