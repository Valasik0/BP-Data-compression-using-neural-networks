import numpy as np
from collections import defaultdict
class KthEntropyCalculator:
    def __init__(self, text, k):
        self.text = text
        self.k = k
        self.k_tuples_count, self.followers_count = self.k_tuples_and_followers()

    def k_tuples_and_followers(self):
        k_tuples_count = defaultdict(int)
        followers_count = defaultdict(int)

        for i in range(len(self.text) - self.k):
            k_tuple = self.text[i:i + self.k]
            next_char = self.text[i + self.k]

            k_tuples_count[k_tuple] += 1
            followers_count[(k_tuple, next_char)] += 1

        k_tuple = self.text[-self.k:]
        k_tuples_count[k_tuple] += 1

        return k_tuples_count, followers_count

    def calculate_kth_entropy(self):
        entropy = 0
        n = len(self.text)
        log2 = np.log2

        for (w, x), f_wx in self.followers_count.items():
            f_w = self.k_tuples_count[w]
            entropy -= (f_wx / n) * log2(f_wx / f_w)

        return entropy