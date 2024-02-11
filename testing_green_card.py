import matplotlib.pyplot as plt

from language import Language
from priors import Priors
from agents import RationalAgent, _fix_nan, _normalize
import torch
import numpy as np
import pandas as pd
l = Language("languages/greencard_3_worlds.json")
p = Priors(l)
a = RationalAgent(l, p, alpha=100, cost_multiplier=0.02, qud_utility=True)
spkr = a.pretty_print(a.speaker_table, ["QUD", "Context", "Utterance", "World"], ["Utterance"])
utils = a.pretty_print(a.make_utilities(), ["QUD", "Context", "Utterance", "World"], ["Utterance"])
pragmatic_listener_worlds = a.pretty_print(torch.sum(a.pragmatic_listener_table.nan_to_num(), dim=1), ["QUD", "Utterance", "World"], output_space=["World"])
pragmatic_listener_contexts = a.pretty_print(torch.sum(a.pragmatic_listener_table.nan_to_num(), dim=3), ["QUD", "Context", "Utterance"], output_space=["Context"])




import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(1, 5)
f_ax1 = fig.add_subplot(gs[0, :2])
f_ax2 = fig.add_subplot(gs[0, 2:])
QUD = "need visa?"

g = sns.barplot(pragmatic_listener_worlds.stack().loc[QUD].reset_index(),
                x="Utterance",
                hue="World",
                y="W | Q, U",
                order=["US", "not US", "green card", "not green card", "_"],
                ax=f_ax1)
g.set_ylabel("P(w | u, need visa?)")
sns.move_legend(g, "upper left")
# plt.savefig("figures/greencard_world_posterior.pdf")


context_distr = pragmatic_listener_contexts.stack().loc[QUD].reset_index()
context_distr["Context"] = context_distr["Context"].apply(
    lambda x: {
        "111": "any",
        "110": "US or green card",
        "101": "US, or non-US/non-green card",
        "100": "US only",
        "011": "non-US",
        "010": "green card only",
        "001": "non-US/non-green card only"
    }[x]
)
g = sns.barplot(context_distr,
                x="Utterance",
                hue="Context",
                y="C | Q, U",
                order=["US", "not US", "green card", "not green card", "_"],
                ax=f_ax2)
g.set_ylabel("P(C | u, need visa?)")
sns.move_legend(g, "upper left")
pass

plt.clf()
x = utils.loc["need visa?", :, "no green card, no visa"][[("U | Q, C, W", "needs visa"), ("U | Q, C, W", "not green card")]]\
    .dropna().stack().reset_index()
x["Context"] = x["Context"].apply(
    lambda x: {
        "111": "any",
        "110": "US or green card",
        "101": "US, or non-US/non-green card",
        "100": "US only",
        "011": "non-US",
        "010": "green card only",
        "001": "non-US/non-green card only"
    }[x]
)
g = sns.barplot(x,
                x="Context",
                y="U | Q, C, W",
                hue="Utterance")
