import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
import matplotlib.image as mpimg

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""
        ###### START CODE HERE ######
        self.indoor = mpimg.imread('indoor.png')
        self.outdoor = mpimg.imread('outdoor.png')
        ###### END CODE HERE ######
        pass
    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        
        ###### START CODE HERE ######
        # R_indoor = self.indoor[:,:,0]
        # G_indoor = self.indoor[:,:,1]
        # B_indoor = self.indoor[:,:,2]
        # R_outdoor = self.outdoor[:,:,0]
        # G_outdoor = self.outdoor[:,:,1]
        # B_outdoor = self.outdoor[:,:,2]
        # plt.imshow(R_indoor)
        # plt.show()
        # plt.imsave('red-channel-indoor.png', R_indoor)
        # plt.imshow(G_indoor)
        # plt.show()
        # plt.imsave('green-channel-indoor.png', G_indoor)
        # plt.imshow(B_indoor)
        # plt.show()
        # plt.imsave('blue-channel-indoor.png', B_indoor)
        # plt.imshow(R_outdoor)
        # plt.show()
        # plt.imsave('red-channel-outdoor.png', R_outdoor)
        # plt.imshow(G_outdoor)
        # plt.show()
        # plt.imsave('greeb-channel-outdoor.png', G_outdoor)
        # plt.imshow(B_outdoor)
        # plt.show()
        # plt.imsave('blue-channel-outdoor.png', B_outdoor)

        # lab_img_indoor = cv2.cvtColor(self.indoor, cv2.COLOR_RGB2LAB)
        # lab_img_outdoor = cv2.cvtColor(self.outdoor, cv2.COLOR_RGB2LAB)
        # L_indoor = lab_img_indoor[:,:,0]
        # A_indoor = lab_img_indoor[:,:,1]
        # B_indoor = lab_img_indoor[:,:,2]
        # L_outdoor = lab_img_outdoor[:,:,0]
        # A_outdoor = lab_img_outdoor[:,:,1]
        # B_outdoor = lab_img_outdoor[:,:,2]
        # plt.imshow(L_indoor)
        # plt.show()
        # plt.imsave('L-channel-indoor.png', L_indoor)
        # plt.imshow(A_indoor)
        # plt.show()
        # plt.imsave('A-channel-indoor.png', A_indoor)
        # plt.imshow(B_indoor)
        # plt.show()
        # plt.imsave('B-channel-indoor.png', B_indoor)
        # plt.imshow(L_outdoor)
        # plt.show()
        # plt.imsave('L-channel-outdoor.png', L_outdoor)
        # plt.imshow(A_outdoor)
        # plt.show()
        # plt.imsave('A-channel-outdoor.png', A_outdoor)
        # plt.imshow(B_outdoor)
        # plt.show()
        # plt.imsave('B-channel-outdoor.png', B_outdoor)

        ###### END CODE HERE ######
        pass

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
        
        img = io.imread('inputPS1Q4.jpg') 
        img = img / 255.0
        
        ###### START CODE HERE ######
        shape = np.shape(img)
        hsv_img = np.zeros_like(img)
        print(shape)
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                pixel = img[i,j,:]
                h, s, v = self.rgb_to_hsv_onepixel(pixel[0], pixel[1], pixel[2])
                hsv_img[i,j,:] = [h,s,v]
        plt.imshow(hsv_img, cmap='hsv')
        plt.show()
        plt.imsave('hsv.png', hsv_img, cmap='hsv')
        return hsv_img

        ###### END CODE HERE ######
        pass
    
        ###### return HSV ######
        

    def rgb_to_hsv_onepixel(self, r, g, b):
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx-mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = ((g-b)/df)
        elif mx == g:
            h = ((b-r)/df) + 2
        elif mx == b:
            h = ((r-g)/df) + 4
        if mx == 0:
            s = 0
        else:
            s = (df/mx)
        v = mx
        return h, s, v
        
if __name__ == '__main__':
    
    p4 = Prob4()
    
    p4.prob_4_1()

    HSV = p4.prob_4_2()





