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
    @staticmethod 
    def simulate(dprime, criteriaList, signalCount, noiseCount):
        sdtList = []
        for i in range(len(criteriaList)):
            criterion = criteriaList[i]
            k = criterion + (dprime/2)
            hitrate = 1 - scipy.stats.norm.cdf(k - dprime)
            falsealarmrate = 1 - scipy.stats.norm.cdf(k)
            hits = np.random.binomial(signalCount, hitrate)
            misses = signalCount - hits
            false_alarms = np.random.binomial(noiseCount, falsealarmrate)
            correct_rejections = noiseCount - false_alarms
            sdtList.append(SignalDetection(hits, misses, false_alarms, correct_rejections))
        return sdtList

    def plot_sdt(self):
        # Set up x values
        x = np.linspace(-4, 4, 1000)
    
        # Compute y values for noise and signal curves
        y_N = scipy.stats.norm.pdf(x, loc = 0, scale = 1) #norm dist with mean 0 and variance 1
        y_S = scipy.stats.norm.pdf(x, loc = self.d_prime, scale = 1) #norm dist with mean d' and varance 1
        c = self.d_prime/2 #optimal threshold

        #calculate tops of x and y
        Ntop_y = np.max(y_N)
        Nstop_x = x[np.argmax(y_N)]
        Stop_y = np.max(y_S)
        Stop_x = x[np.argmax(y_S)]
    

        # Plot curves and add annotations
        plt.plot(x, y_N, label="Noise") # plot N curve
        plt.plot(x, y_S, label="Signal") # plot S curve
        plt.axvline((self.d_prime/2)+ c,label = 'threshold', color='k', linestyle='--') # plot threshold line C
        plt.plot ([Nstop_x, Stop_x ],[ Ntop_y, Stop_y], label = "d'", linestyle = '-') #plot dprime line 
        plt.ylim(ymin=0)
        plt.xlabel('Decision Variable')
        plt.ylabel('Probability')
        plt.title('Signal detection theory')
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_roc(sdtList):
        plt.figure()
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("Receiver Operating Characteristic Curve")
        if isinstance(sdtList, list):
            for i in range(len(sdtList)):
                sdt = sdtList[i]
                plt.plot(sdt.farate(), sdt.hitrate(), 'o', color = 'black')
        x, y = np.linspace(0,1,100), np.linspace(0,1,100)
        plt.plot(x,y, '--', color = 'black')
        plt.grid()

    @staticmethod
    def rocCurve(falseAlarmRate, a):
        return scipy.stats.norm.cdf(a + scipy.stats.norm.ppf((falseAlarmRate)))
    
    def nLogLikelihood(self, hit_rate, false_alarm_rate):
        return -((self.hits * np.log(hit_rate)) + 
        (self.misses * np.log(1-hit_rate)) + 
        (self.falseAlarms * np.log(false_alarm_rate)) + 
        (self.correctRejections * np.log(1-false_alarm_rate)))

    @staticmethod
    def fit_roc(sdtList):
        SignalDetection.plot_roc(sdtList)
        a = 0
        minimize = scipy.optimize.minimize(fun = SignalDetection.rocLoss, x0 = a, method = 'BFGS', args = (sdtList))
        loss = []
        for i in range(0,100,1):
            loss.append((SignalDetection.rocCurve(i/100, float(minimize.x))))
        plt.plot(np.linspace(0,1,100), loss, '-', color = 'r')
        aHat = minimize.x
        return float(aHat)
    
    @staticmethod
    def rocLoss(a, sdtList):
        total_loss = 0
        for i in range(len(sdtList)):
           sdt = sdtList[i]
           farate = sdt.farate()
           predicted_hr = sdt.rocCurve(farate, a)
           loss_i = sdt.nLogLikelihood(predicted_hr, farate)
           total_loss += loss_i
        return total_loss 
    
   
sdtList = SignalDetection.simulate(1, [-1, 0, 1], 1e7, 1e7)
SignalDetection.fit_roc(sdtList)
plt.show()
  




class TestSignalDetection(unittest.TestCase):
    """
    Test suite for SignalDetection class.
    """

    def test_d_prime_zero(self):
        """
        Test d-prime calculation when hits and false alarms are 0.
        """
        sd   = SignalDetection(15, 5, 15, 5)
        expected = 0
        obtained = sd.d_prime()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_d_prime_nonzero(self):
        """
        Test d-prime calculation when hits and false alarms are nonzero.
        """
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        obtained = sd.d_prime()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_zero(self):
        """
        Test criterion calculation when hits and false alarms are both 0.
        """
        sd   = SignalDetection(5, 5, 5, 5)
        expected = 0
        obtained = sd.criterion()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_nonzero(self):
        """
        Test criterion calculation when hits and false alarms are nonzero.
        """
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.463918426665941
        obtained = sd.criterion()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_addition(self):
        """
        Test addition of two SignalDetection objects.
        """
        sd = SignalDetection(1, 1, 2, 1) + SignalDetection(2, 1, 1, 3)
        expected = SignalDetection(3, 2, 3, 4).criterion()
        obtained = sd.criterion()
        self.assertEqual(obtained, expected)

    def test_multiplication(self):
        """
        Test multiplication of a SignalDetection object with a scalar.
        """
        sd = SignalDetection(1, 2, 3, 1) * 4
        expected = SignalDetection(4, 8, 12, 4).criterion()
        obtained = sd.criterion()
        self.assertEqual(obtained, expected)

    def test_simulate_single_criterion(self):
        """
        Test SignalDetection.simulate method with a single criterion value.
        """
        dPrime       = 1.5
        criteriaList = [0]
        signalCount  = 1000
        noiseCount   = 1000
        
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 1)
        sdt = sdtList[0]
        
        self.assertEqual(sdt.hits             , sdtList[0].hits)
        self.assertEqual(sdt.misses           , sdtList[0].misses)
        self.assertEqual(sdt.falseAlarms      , sdtList[0].falseAlarms)
        self.assertEqual(sdt.correctRejections, sdtList[0].correctRejections)

    def test_simulate_multiple_criteria(self):
        """
        Test SignalDetection.simulate method with multiple criterion values.
        """
        dPrime       = 1.5
        criteriaList = [-0.5, 0, 0.5]
        signalCount  = 1000
        noiseCount   = 1000
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 3)
        for sdt in sdtList:
            self.assertLessEqual (sdt.hits              ,  signalCount)
            self.assertLessEqual (sdt.misses            ,  signalCount)
            self.assertLessEqual (sdt.falseAlarms       ,  noiseCount)
            self.assertLessEqual (sdt.correctRejections ,  noiseCount)
   
    def test_nLogLikelihood(self):
        """
        Test case to verify nLogLikelihood calculation for a SignalDetection object.
        """
        sdt = SignalDetection(10, 5, 3, 12)
        hit_rate = 0.5
        false_alarm_rate = 0.2
        expected_nll = - (10 * np.log(hit_rate) +
                           5 * np.log(1-hit_rate) +
                           3 * np.log(false_alarm_rate) +
                          12 * np.log(1-false_alarm_rate))
        self.assertAlmostEqual(sdt.nLogLikelihood(hit_rate, false_alarm_rate),
                               expected_nll, places=6)
        
    def test_rocLoss(self):
        """
        Test case to verify rocLoss calculation for a list of SignalDetection objects.
        """
        sdtList = [
            SignalDetection( 8, 2, 1, 9),
            SignalDetection(14, 1, 2, 8),
            SignalDetection(10, 3, 1, 9),
            SignalDetection(11, 2, 2, 8),
        ]
        a = 0
        expected = 99.3884
        self.assertAlmostEqual(SignalDetection.rocLoss(a, sdtList), expected, places=4)
        
    def test_integration(self):
         """
         Test case to verify integration of SignalDetection simulation and ROC fitting.
         """
         dPrime  = 1
         sdtList = SignalDetection.simulate(dPrime, [-1, 0, 1], 1e7, 1e7)
         aHat    = SignalDetection.fit_roc(sdtList)
         self.assertAlmostEqual(aHat, dPrime, places=2)
         plt.close()
        
if __name__ == '__main__':
    unittest.main()


