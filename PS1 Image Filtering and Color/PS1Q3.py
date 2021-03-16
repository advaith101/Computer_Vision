import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
import matplotlib.image as mpimg

class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        ###### START CODE HERE ######
        self.img = mpimg.imread('inputPS1Q3.jpg')

        ###### END CODE HERE ######
        pass
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        return gray
        ###### END CODE HERE ######
        pass
    
        ###### return gray ######
        
    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        swapImg = self.img.copy()
        red = self.img[:,:,0].copy()
        green = self.img[:,:,1].copy()
        swapImg[:,:,0] = green
        swapImg[:,:,1] = red
        plt.imshow(swapImg)
        plt.show()
        plt.imsave('q3-swapped.png', swapImg)
        return swapImg
        ###### END CODE HERE ######
        pass
    
        ###### return swapImg ######
    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayImg = self.rgb2gray(self.img)
        plt.imshow(grayImg, cmap='gray', vmin=0, vmax=255)
        plt.show()
        plt.imsave('q3-grayscale.png', grayImg, cmap='gray')
        return grayImg
        ###### END CODE HERE ######
        pass
    
        ###### return grayImg ######
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayscale = self.rgb2gray(self.img)
        negativeImg = np.full((np.shape(grayscale)[0],np.shape(grayscale)[1]), 255) - grayscale
        plt.imshow(negativeImg, cmap='gray', vmin=0, vmax=255)
        plt.show()
        plt.imsave('q3-grayscale-negative.png', negativeImg, cmap='gray')
        return negativeImg
        ###### END CODE HERE ######
        pass
    
        ###### return negativeImg ######
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayscale = self.rgb2gray(self.img)
        mirrorImg = np.fliplr(grayscale)
        plt.imshow(mirrorImg, cmap='gray', vmin=0, vmax=255)
        plt.show()
        plt.imsave('q3-grayscale-mirrored.png', mirrorImg, cmap='gray')
        return mirrorImg
        ###### END CODE HERE ######
        pass
    
        ###### return mirrorImg ######
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayscale = (self.rgb2gray(self.img)).astype(np.double)
        mirrorImg = (np.fliplr(grayscale)).astype(np.double)
        avgImg = ((grayscale + mirrorImg)/2).astype(np.uint8)
        # print(avgImg)
        plt.imshow(avgImg, cmap='gray', vmin=0, vmax=255)
        plt.show()
        plt.imsave('q3-grayscale-mirror-avg.png', avgImg, cmap='gray')
        return avgImg
        ###### END CODE HERE ######
        pass
    
        ###### return avgImg ######
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            addNoiseImg: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayscale = (self.rgb2gray(self.img)).astype(np.double)
        print("grayscaleShape: {} \n grayscale: {}".format(np.shape(grayscale),grayscale))
        N = (np.random.randint(low=0, high=255, size=(np.shape(grayscale)[0], np.shape(grayscale)[1]))).astype(np.double)
        print("noise: {}".format(N))
        np.save('noise.npy', N)
        added = N + grayscale
        print("added: {}".format(added))
        addNoiseImgnotypecast = (np.clip(added, 0, 255))
        print("max: {}, min: {}, shape: {}". format(np.max(addNoiseImgnotypecast), np.min(addNoiseImgnotypecast), np.shape(addNoiseImgnotypecast)))
        print("not type casted: {}".format(addNoiseImgnotypecast))
        addNoiseImg = (np.where(added>255, 255, added)).astype(np.uint8)
        print("final: {}".format(addNoiseImg))
        plt.imshow(addNoiseImg, cmap='gray', vmin=0, vmax=255)
        plt.show()
        plt.imsave('q3-grayscale-noise.png', addNoiseImg, cmap='gray')
        return addNoiseImg
        ###### END CODE HERE ######
        pass
    
        ###### return addNoiseImg ######
        
        
if __name__ == '__main__': 
    
    p3 = Prob3()
    
    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    addNoiseImg = p3.prob_3_6()