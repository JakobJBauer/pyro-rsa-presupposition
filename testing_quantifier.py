from language import Language
from priors import Priors
from agents import RationalAgent, _fix_nan, _normalize
import torch
import numpy as np
import pandas as pd
# l = Language("languages/greencard_4_worlds.json")


# Testing quantifier language

l = Language("languages/quantifier.json")
p = Priors(l)
a = RationalAgent(l, p, alpha=500, cost_multiplier=0.02, qud_utility=False)
spkr = a.pretty_print(a.speaker_table, ["QUD", "Context", "Utterance", "World"], ["Utterance"])
utils = a.pretty_print(_fix_nan(a.make_utilities()), ["QUD", "Context", "Utterance", "World"], ["Utterance"])
pragmatic_listener_worlds = a.pretty_print(torch.sum(a.pragmatic_listener_table.nan_to_num(), dim=1), ["QUD", "Utterance", "World"], output_space=["World"])
pragmatic_listener_contexts = a.pretty_print(torch.sum(a.pragmatic_listener_table.nan_to_num(), dim=3), ["QUD", "Context", "Utterance"], output_space=["Context"])

# Look at inferred distribution over contexts
context_distr = torch.sum(a.pragmatic_listener_table.nan_to_num(), dim=3)[0]
prod = l.contexts.expand(l.n_utterances, -1, -1) * context_distr.expand(l.n_worlds, -1, -1).transpose(0, 2)
inferred_worlds_prior = torch.sum(_normalize(prod, dim=(1, 2)), dim=1)

inferred_non_us_prior = torch.sum(
    l.quds[l.qud_to_idx("How many non-US?")].expand(l.n_utterances, -1, -1) *
    inferred_worlds_prior.expand(l.n_alternatives(l.qud_to_idx("How many non-US?")), -1, -1).transpose(0, 1),
    dim=2)

inferred_non_us_prior = a.pretty_print(inferred_non_us_prior, ["Utterance", "Alternative"], ["Alternative"], qud_idx=1)

# Look at posterior over worlds grouped by QUD alternatives
world_distr = torch.sum(a.pragmatic_listener_table.nan_to_num(), dim=1)[0]
qud_distr = torch.sum(
    l.quds[l.qud_to_idx("How many non-US?")].expand(l.n_utterances, -1, -1) *
    world_distr.expand(l.n_alternatives(l.qud_to_idx("How many non-US?")), -1, -1).transpose(0, 1),
    dim=2)
posterior = a.pretty_print(qud_distr, ["Utterance", "Alternative"], ["Alternative"], qud_idx=1)
pass


import seaborn as sns

g = sns.pointplot(data=inferred_non_us_prior.loc[["none green card"]].stack().reset_index(), x="Alternative", y="A | U", order=[str(i) for i in range(11)])
g.set_ylabel("Probability according to the posterior context distribution")

pass