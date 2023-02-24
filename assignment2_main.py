import unittest
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

class SignalDetection: 
    def __init__ (self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses 
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections 

    def hitrate(self):
        return self.hits/(self.hits + self.misses)
    
    def farate(self):
        return self.falseAlarms/(self.falseAlarms+self.correctRejections)
    
    def d_prime(self):
        return scipy.stats.norm.ppf(self.hitrate()) - scipy.stats.norm.ppf(self.farate())
    
    def criterion(self):
        return  -0.5 * (scipy.stats.norm.ppf(self.hitrate())+ scipy.stats.norm.ppf(self.farate()))
    #operator overloading
    def __add__(self, other):
         hits = self.hits + other.hits
         misses = self.misses + other.misses 
         falseAlarms = self.falseAlarms + other.falseAlarms
         correctRejections = self.correctRejections + other.correctRejections
         return SignalDetection(hits, misses, falseAlarms, correctRejections) 
    
    def __mul__(self, scalar):
        hits = self.hits * scalar 
        misses = self.misses * scalar
        falseAlarms = self.falseAlarms * scalar
        correctRejections = self.correctRejections * scalar
        return SignalDetection(hits, misses, falseAlarms, correctRejections)

    def plot_roc(self):
        # Calculate the hit rate and false alarm rate for each object
        hitrate = self.hitrate()
        falsealarm = self.farate()
    
        #coordinates to make the line go from (0,0) - (falsealarm,hitrate) - (1,1)
        x_coords = [0.0, falsealarm, 1.0]
        y_coords = [0.0, hitrate, 1.0]

        #plotting and design
        plt.plot(x_coords, y_coords, 'bo-', label = 'ROC curve')
        plt.plot([0, 1], [0, 1], 'k-', label = 'reference line') # diagonal line for reference
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Alarm Rate')
        plt.ylabel('Hit Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()


#unit test
 
class TestSignalDetection(unittest.TestCase):
    def test_d_prime_zero(self):
        sd   = SignalDetection(15, 5, 15, 5)
        expected = 0
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_d_prime_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_zero(self):
        sd   = SignalDetection(5, 5, 5, 5)
        # Calculate expected criterion
        expected = 0
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        # Calculate expected criterion
        expected = -0.463918426665941
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=10)
    
    def test_for_corruption(self):
        sd = SignalDetection(15, 5, 15, 5)
        obtained1 = sd.d_prime()
        sd.hits = 9
        obtained2 = sd.d_prime()
	#compare the 2 d primes obtained 
        self.assertNotEqual(obtained1,obtained2)

    def test_addition(self):
        sd = SignalDetection(1, 1, 2, 1) + SignalDetection(2, 1, 1, 3)
        expected = SignalDetection(3, 2, 3, 4).criterion()
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertEqual(obtained, expected)

    def test_multiplication(self):
        sd = SignalDetection(1, 2, 3, 1) * 4
        expected = SignalDetection(4, 8, 12, 4).criterion()
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertEqual(obtained, expected)

if __name__ == '__main__':
    unittest.main()

