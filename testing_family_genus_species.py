from language import Language
from priors import Priors
from agents import RationalAgent, _fix_nan, _normalize
import torch
import numpy as np
import pandas as pd
import seaborn as sns
# l = Language("languages/greencard_4_worlds.json")


# Testing quantifier language

l = Language("languages/olympic_sprinter_simple.json")
p = Priors(l)
a = RationalAgent(l, p, alpha=10e2, cost_multiplier=0.02, qud_utility=True)
context_probs = torch.sum(a.pragmatic_listener_table[0].nan_to_num(), dim=2)
t = a.pretty_print(context_probs, ["Context", "Utterance"], ["Context"])
prod = l.contexts.expand(l.n_utterances, -1, -1) * l.contexts.expand(l.n_utterances, -1, -1) * context_probs.expand(l.n_worlds, -1, -1).transpose(0, 2) * p.world_prior()
inferred_worlds_prior = torch.sum(_normalize(prod, dim=(1, 2)), dim=1)
print("athlete | ", inferred_worlds_prior)
inferred_worlds_prior = a.pretty_print(inferred_worlds_prior, ["Utterance", "World"], ["World"])
# print(inferred_worlds_prior.to_string())

# #Plotting probability over contexts distributed by world
df = inferred_worlds_prior.stack().reset_index()
g = sns.barplot(df,
                x="Utterance",
                hue="World",
                y="W | U",
                order=["Olympic sprinter", "Not olympic sprinter", "Runner", "Not runner", "Athlete", "Not athlete", "_"])
g.set_ylabel("P(C | U) distributed by world")

pass

# Plot Listener posterior
l_post = torch.sum(a.pragmatic_listener_table[0].nan_to_num(), dim=0)
t = a.pretty_print(l_post, ["Utterance", "World"], ["World"])
df = t.stack().reset_index()
g = sns.barplot(df,
                x="Utterance",
                hue="World",
                y="W | U",
                order=["Olympic sprinter", "Not olympic sprinter", "Runner", "Not runner", "Athlete", "Not athlete", "_"])
pass