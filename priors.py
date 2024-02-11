import numpy as np
import torch
from utils import _normalize

class Priors:
    def __init__(self, language):
        self.language = language

    def world_prior(self):
        """Uniform prior"""
        if "world_prior" in self.language.language_specs:
            return torch.Tensor(list(self.language.language_specs["world_prior"].values()))
        return torch.empty(self.language.n_worlds).fill_(1/self.language.n_worlds)

    def utterance_prior(self):
        """Uniform prior"""
        return torch.empty(len(self.language.utterances)).fill_(1/len(self.language.utterances))

    def context_prior(self):
        """uniform over contexts (this could be modified)"""
        # if "world_prior" in self.language.language_specs:
        #     # You shouldn't assign high probability to contexts with only highly unlikely world
        #     _normalize(torch.mean(self.language.contexts * self.world_prior(), dim=1), dim=0)
        return torch.empty(self.language.n_contexts).fill_(1/self.language.n_contexts)

    def world_prior_gen(self):
        return np.random.choice(self.language.worlds)

    def utterance_prior_gen(self):
        return np.random.choice(self.language.utterances)

    def context_prior_gen(self, kappa):
        for _ in range(512):
            context = torch.zeros(self.language.n_worlds).uniform_(0, 1) < kappa
            if torch.is_nonzero(context):
                return context
        raise RuntimeError("Context prior doesn't generate valid belief states")