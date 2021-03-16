import numpy as np
import matplotlib.pyplot as plt

class Prob2():
    def __init__(self):
        """Load inputAPS1Q2.npy here as a class variable A."""
        ###### START CODE HERE ######
        self.A = np.load('inputAPS1Q2.npy')
        ###### END CODE HERE ######
        pass
        
    def prob_2_1(self):
        """Do plotting of intensities of A in decreasing value."""
        ###### START CODE HERE ######
        sorted_by_intensity_decreasing = np.sort(self.A)[::-1]
        plt.plot(sorted_by_intensity_decreasing)
        plt.show()
        ###### END CODE HERE ######
        pass
    
    def prob_2_2(self):
        """Display histogram of A's intensities with 20 bins here."""
        ###### START CODE HERE ######
        hist = np.histogram(self.A, 20)
        plt.hist(self.A, bins=20)
        plt.show()
        ###### END CODE HERE ######
        pass
    
    def prob_2_3(self):
        """
        Create a new matrix X that consists of the bottom left quadrant of A here.
        Returns:
            X: bottom left quadrant of A which is of size 50 x 50
        """
        ###### START CODE HERE ######
        # h = len(self.A)
        # w = len(self.A[1])
        # X =  [self.A[i][:h // 2] for i in range(w // 2, w)]
        # a, b, c, d = self.A[:50, :50], self.A[:50, 50:], self.A[:50, 50:], self.A[50:, 50:]
        # # a, b, c, d = np.split()
        # lower_half = np.hsplit(np.vsplit(self.A, 2)[1], 2)
        # lower_left = lower_half[0]
        # arr = np.array([[2,3,4,5],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
        # X = arr[2:, :2]
        X = self.A[50:, :50]
        # print([[8,9],[12,13]])
        print(X)
        print(np.shape(X))
        plt.imshow(X, interpolation='none')
        plt.show()
        return X
        ###### END CODE HERE ######
            
        ###### return X ###### 
    
    def prob_2_4(self):
        """Create a new matrix Y, which is the same as A, but with A’s mean intensity value subtracted from each pixel.
        Returns:
            Y: A with A's mean intensity subtracted from each pixel. Output Y is of size 100 x 100.
        """
        ###### START CODE HERE ######

        Y = self.A - (np.full(np.shape(self.A), np.mean(self.A)))
        plt.imshow(Y)
        plt.show()
        return Y
        ###### END CODE HERE ######
        
    
        ###### return Y ######
    
    def prob_2_5(self):
        """
        Create your threshholded A i.e Z here.
        Returns:
            Z: A with only red pixels when the original value of the pixel is above the threshhold. Output Z is of size 100 x 100.
        """
        ###### START CODE HERE ######
        Z = np.zeros((100,100,3))
        red_pixels = np.where(self.A > np.mean(self.A), 1.0, 0.0)
        Z[:,:,0] = red_pixels
        # Z = Z.astype("uint8")
        plt.imshow(Z)
        plt.show()
        return Z
        ###### END CODE HERE ######
     
    
        ###### return Z ######
        
        
        
if __name__ == '__main__':
    
    p2 = Prob2()
    
    p2.prob_2_1()
    p2.prob_2_2()
    
    X = p2.prob_2_3()
    Y = p2.prob_2_4()
    Z = p2.prob_2_5()