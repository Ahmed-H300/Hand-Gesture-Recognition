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
    img = cv2.imread('test.jpg')
    k, d = sift.compute(img)
    img_keypoints = sift.draw_descriptors(k, img)
    # save the image with the keypoints on it
    cv2.imwrite('test_sift.jpg', img_keypoints)
