import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats

class Metropolis:
    def __init__(self, logTarget, initialState):
        self.logTarget = logTarget
        self.currentState = initialState
        self.samples = []
        self.sd = 1
        
    def __accept(self, proposal):
        logAlpha = min(0,self.logTarget(proposal) - self.logTarget(self.currentState))
        logU = np.log(np.random.uniform())
        if (logU < logAlpha):
            self.currentState = proposal
            return True
        else:
            return False
        
    def adapt(self, blockLengths):
        for k in blockLengths:
            acceptances = 0
            for n in range(k):
                proposal = np.random.normal(loc=self.currentState, scale=self.sd)
                if self.__accept(proposal):
                    acceptances += 1
            acceptanceRate = acceptances / n
            self.sd = self.sd * (acceptanceRate/0.4)**1.1
        return self
        
    def sample(self, nSamples):
        for n in range(nSamples):
            proposal = np.random.normal(loc=self.currentState, scale=self.sd)
            self.__accept(proposal)
            self.samples.append(self.currentState)
        return self
    
    def summary(self):
        samples = np.array(self.samples)
        mean = np.mean(self.samples)
        c025 = np.percentile(self.samples, 2.5)
        c975 = np.percentile(self.samples, 97.)
    
    # Return summary statistics as a dictionary
        return {'mean': mean, 'c025': c025, 'c975': c975}
    

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
            falseAlarms = np.random.binomial(noiseCount, falsealarmrate)
            correctRejections = noiseCount - falseAlarms
            sdtList.append(SignalDetection(hits, misses, falseAlarms, correctRejections))
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
        #plt.figure()
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


import scipy.stats  
def fit_roc_bayesian(sdtList):

    # Define the log-likelihood function to optimize
    def loglik(a):
        return -SignalDetection.rocLoss(a, sdtList) + scipy.stats.norm.logpdf(a, loc=0, scale=10)

    # Create a Metropolis sampler object and adapt it to the target distribution
    sampler = Metropolis(logTarget=loglik, initialState=0)
    sampler = sampler.adapt(blockLengths=[2000]*3)

    # Sample from the target distribution
    sampler = sampler.sample(nSamples=4000)

    # Compute the summary statistics of the samples
    result = sampler.summary()

    # Print the estimated value of the parameter a and its credible interval
    print(
        f"Estimated a: {result['mean']} ({result['c025']}, {result['c975']})")

    # Create a mosaic plot with four subplots
    fig, axes = plt.subplot_mosaic(
        [["ROC curve", "ROC curve", "traceplot"],
         ["ROC curve", "ROC curve", "histogram"]],
        constrained_layout=True
    )

    # Plot the ROC curve of the SDT data
    plt.sca(axes["ROC curve"])
    SignalDetection.plot_roc(sdtList=sdtList)

    # Compute the ROC curve for the estimated value of a and plot it
    xaxis = np.arange(start=0.00,
                      stop=1.00,
                      step=0.01)

    plt.plot(xaxis, SignalDetection.rocCurve(xaxis, result['mean']), 'r-')

    # Shade the area between the lower and upper bounds of the credible interval
    plt.fill_between(x=xaxis,
                     y1=SignalDetection.rocCurve(xaxis, result['c025']),
                     y2=SignalDetection.rocCurve(xaxis, result['c975']),
                     facecolor='r',
                     alpha=0.1)

    # Plot the trace of the sampler
    plt.sca(axes["traceplot"])
    plt.plot(sampler.samples)
    plt.xlabel('iteration')
    plt.ylabel('a')
    plt.title('Trace plot')

    # Plot the histogram of the samples
    plt.sca(axes["histogram"])
    plt.hist(sampler.samples,
             bins=51,
             density=True)
    plt.xlabel('a')
    plt.ylabel('density')
    plt.title('Histogram')

    # Show the plot
    plt.show()


# Define the number of SDT trials and generate a simulated dataset
sdtList = SignalDetection.simulate(dprime=1,
                                   criteriaList=[-1, 0, 1],
                                   signalCount=40,
                                   noiseCount=40)

# Fit the ROC curve to the simulated dataset
fit_roc_bayesian(sdtList)

import unittest

class TestSignalDetection(unittest.TestCase):
    def test_simulate(self):
        # Test with a single criterion value
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

        # Test with multiple criterion values
        dPrime       = 1.5
        criteriaList = [-0.5, 0, 0.5]
        signalCount  = 1000
        noiseCount   = 1000
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 3)
        for sdt in sdtList:
            self.assertLessEqual    (sdt.hits              ,  signalCount)
            self.assertLessEqual    (sdt.misses            ,  signalCount)
            self.assertLessEqual    (sdt.falseAlarms       ,  noiseCount)
            self.assertLessEqual    (sdt.correctRejections ,  noiseCount)

    def test_nLogLikelihood(self):
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
        sdtList = [
            SignalDetection( 8, 2, 1, 9),
            SignalDetection(14, 1, 2, 8),
            SignalDetection(10, 3, 1, 9),
            SignalDetection(11, 2, 2, 8),
        ]
        a = 0
        expected = 99.3884206555698
        self.assertAlmostEqual(SignalDetection.rocLoss(a, sdtList), expected, places=4)

    def test_integration(self):
        dPrime = 1
        sdtList = SignalDetection.simulate(dPrime, [-1, 0, 1], 1e7, 1e7)
        aHat = SignalDetection.fit_roc(sdtList)
        self.assertAlmostEqual(aHat, dPrime, places=2)
        plt.close()

if __name__ == '__main__':
    unittest.main() # for jupyter
