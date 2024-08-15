import numpy as np
# import matplotlib.pyplot as plt


class MultiArmedBandit:
    def __init__(self, k: int = 10) -> None:
        self.k = k


class NormalMultiArmedBandit(MultiArmedBandit):
    def __init__(self, k: int,
                 means: np.array,
                 sigmas: np.array
                 ) -> None:
        super().__init__(k)
        assert len(means) == k, f"len(means) should be equal to k, but len(means) = {len(means)} and k = {k}"
        self.means = means
        assert len(sigmas) == k, f"len(sigmas) should be equal to k, but len(sigmas) = {len(sigmas)} and k = {k}"
        assert (sigmas < 0).sum() == 0, "all values in sigmas must be non negative"
        self.sigmas = sigmas

    def run(self, action: int):
        return np.random.normal(loc=self.means[action],
                                scale=self.sigmas[action])

        return None


class BernoulliMultiArmedBandit(MultiArmedBandit):
    def __init__(self, k: int,
                 ps: np.array
                 ) -> None:
        super().__init__(k)
        assert len(ps) == k, f"len(ps) should be equal to k, but len(ps) = {len(ps)} and k = {k}"
        assert (ps < 0).sum() == 0, "all values in ps must be non negative"
        assert (ps > 1).sum() == 0, "all values in ps must be <= 1"
        self.ps = ps

    def run(self, action: int):
        return np.random.binomial(n=1, p=self.ps[action])


class TaskDistribution():
    def __init__(self, k: int = 10) -> None:
        self.k = k

    def sample_task(self):
        return MultiArmedBandit(self.k)


class BernoulliMultiArmedBandit_TaskDistribution(TaskDistribution):
    def __init__(self, k: int = 10, distribution_type: str = "odd", q: float = 0.95) -> None:
        super().__init__(k)
        self.distribution_type = distribution_type
        self.q = q

    def sample_odd_or_even(self, odd: bool = True):
        ps = np.random.uniform(low=0, high=1, size=(self.k,))
        while np.argmax(ps) % 2 != odd:
            ps = np.random.uniform(low=0, high=1, size=(self.k,))
        return BernoulliMultiArmedBandit(self.k, ps)

    def sample_task(self):
        q = self.q
        if self.distribution_type == "odd":
            sample_of_odd_reward = np.random.uniform()
            if sample_of_odd_reward <= q:
                return self.sample_odd_or_even(odd=True)
            else:
                return self.sample_odd_or_even(odd=False)
        elif self.distribution_type == "even":
            sample_of_odd_reward = np.random.uniform()
            if sample_of_odd_reward <= q:
                return self.sample_odd_or_even(odd=False)
            else:
                return self.sample_odd_or_even(odd=True)
        else:
            ps = np.random.uniform(low=0, high=1, size=(self.k,))
            return BernoulliMultiArmedBandit(self.k, ps)
