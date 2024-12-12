import numpy as np

class Weights:
    def __init__(self, efficiency_weight, price_weight, idle_weight, job_age_weight):
        self.efficiency_weight = efficiency_weight
        self.price_weight = price_weight
        self.idle_weight = idle_weight
        self.job_age_weight = job_age_weight

    def __str__(self):
        return f"Weights: efficiency={self.efficiency_weight}, price={self.price_weight}, idle={self.idle_weight}, job_age={self.job_age_weight}. sum={self.sum():.2f})"

    def __repr__(self):
        return self.__str__()

    def sum(self):
        return round(np.sum([self.efficiency_weight, self.price_weight, self.idle_weight, self.job_age_weight]), 2)
