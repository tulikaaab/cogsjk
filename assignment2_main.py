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
