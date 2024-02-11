from language import Language
from priors import Priors
import torch
from functools import reduce
from itertools import product
import pandas as pd
import numpy as np
from utils import _fix_nan, _normalize


class RationalAgent:

    def __init__(
        self,
        language: Language,
        priors: Priors,
        alpha,
        cost_multiplier=0,
        qud_utility=False
    ):
        self.language = language
        self.priors = priors
        self.alpha = alpha
        self.cost_multiplier = cost_multiplier
        self.qud_utility = qud_utility
        self.literal_listener_table = self.make_literal_listener()  # {Q_i: C x A_i x U}
        self.speaker_table = self.make_speaker()
        self.pragmatic_listener_table = self.make_pragmatic_listener()

    def make_literal_listener(self):
        """Returns a probability distribution over qud-alternatives given an utterance and a context
        Dimensions: {Q_i: C x A_i x U} """
        table = {}
        for i, qud in self.language.quds.items():
            n_alts = len(qud)
            # Dimensions: C x A_i x U x W
            table_q = self.priors.world_prior() * \
                      reduce(torch.bitwise_and,
                             [qud.expand(self.language.n_contexts, self.language.n_utterances, -1, -1).transpose(1, 2),
                              self.language.utterances.expand(self.language.n_contexts, n_alts, -1, -1),
                              self.language.contexts.expand(self.language.n_utterances, n_alts, -1, -1).transpose(0, 2)]
                             )
            marginal_table_q = table_q.sum(3)
            table[i] = _normalize(marginal_table_q, 1)
        return table

    def query_literal_listener(self, qud=None, alternative=None, utterance=None, context=None):
        if type(qud) is str:
            qud = self.language._qud_to_idx[qud]
        if type(alternative) is str:
            alternative = self.language.alt_to_idx(qud, alternative)
        if type(utterance) is str:
            utterance = self.language._utt_to_idx[utterance]
        if type(context) is torch.Tensor:
            context = self.language.context_to_idx(context)
        return self.literal_listener_table[qud][context, alternative, utterance]

    def convert_alt_table_to_worlds(self):
        """The listener table uses alternatives, not worlds.
            Also it's a dict, because QUDs have different #s of alts."""
        table = torch.stack([
            torch.matmul(
                self.literal_listener_table[q].transpose(1, 2),
                self.language.quds[q].float()
            )
            for q in self.language.quds.keys()
        ])
        nans = torch.Tensor([float("nan")]).expand(table.size())
        contexts = self.language.contexts.expand(self.language.n_utterances, -1, -1).transpose(0, 1)
        table = torch.where((contexts == 0), nans, table)
        # table = _normalize(table, 3, nans=True)
        return table


    def make_utilities(self, truthfulness_correction=True):
        """Returns a lookup table of utilities for an utterance given a qud, context, and world
        Dimensions: Q x C x U x W """
        literal_listener_world_table = self.convert_alt_table_to_worlds()
        utils = torch.log(literal_listener_world_table)                                                                 # take the negative surprisal
        utils = utils - self.cost_multiplier * self.language.costs.expand_as(utils.transpose(2, 3)).transpose(2, 3)     # subtract the cost
        if truthfulness_correction:
            # Saying false things may get the listener to believe the correct answer to the QUD,
            # but a cooperative speaker still shouldn't do it
            n_infs = torch.Tensor([-np.inf]).expand(utils.size())
            utils = torch.where(
                self.language.utterances.expand(utils.size()),
                utils,
                n_infs
            )
        return utils

    def make_speaker(self):
        """Returns a lookup table of probabilities for an utterance, given a QUD, a context, and a world"""
        utilities = _fix_nan(self.make_utilities())
        # utilities = utilities - \
        #             self.cost_multiplier * self.language.costs.expand_as(utilities.transpose(2, 3)).transpose(2, 3)
        speaker_table = torch.softmax(self.alpha * utilities, dim=2)
        return speaker_table


    def query_speaker(self, qud, context, utterance, world):
        if type(qud) is str:
            qud = self.language._qud_to_idx[qud]
        if type(utterance) is str:
            utterance = self.language._utt_to_idx[utterance]
        if type(context) is torch.Tensor:
            context = self.language.context_to_idx(context)
        return self.speaker_table[qud, context, utterance, world]

    def make_pragmatic_listener(self):
        """Returns a lookup table probabilities for a world & context pair, given a QUD and an utterance"""
        """
        P(w, C | Q, u) \propto P(Q, u | w, C) * P(w, C) 
                             = P(Q, u | w, C) * P(w | C) * P(C)
                             = S(u | Q, w, C) * P(Q | w, C) * P(w | C) * P(C)
        """
        # For first draft, assume all these distributions are uniform. This assumption can be fixed later.
        if self.qud_utility:
            qud_prior = self.make_qud_distribution()\
                .expand(self.language.n_worlds, self.language.n_utterances, -1, -1)\
                .permute(3, 2, 1, 0)
        else:
            qud_prior = torch.ones(self.speaker_table.size())
        context_prior = torch.ones(self.speaker_table.size())

        # This is uniform over all worlds contained in the context.
        world_prior = _normalize(self.language.contexts.int(), 1)\
            .expand(self.language.n_quds, self.language.n_utterances, -1, -1).transpose(1, 2)
        return _normalize(self.speaker_table * qud_prior * context_prior * world_prior, dim=(1, 3), nans=True)

    def make_qud_distribution(self):
        """
        given (a world) and a context, what is the prior probability that a question would have been under discussion?
        This is a very complex question, but we can assume that the probability of a question depends only on its entropy in the context.
        Later, we may want to consider other factors of question utility, such as its answerability in the speaker's belief state,
        or the effort required to formulate and understand an answer.
        """
        rows = []
        for q in self.language.quds.keys():
            alternative_probs = torch.sum(
                self.language.quds[q].expand(self.language.n_contexts, -1, -1) *
                _normalize(self.language.contexts, dim=1).expand(self.language.n_alternatives(q), -1, -1).transpose(0, 1),
                dim=2
            )
            entropy = torch.sum(-1 * alternative_probs * torch.log(alternative_probs).nan_to_num(), dim=1)
            rows.append(entropy)
        entropies = torch.stack(rows).transpose(0, 1)
        costs = self.language.qud_costs.expand_as(entropies) * self.cost_multiplier
        utils = entropies - costs
        return torch.softmax(self.alpha * utils, dim=1)


    def pretty_print(self, tensor, tensor_dims, output_space, dicts=None, qud_idx=None):
        if dicts:
            pass    # TODO: Handle case of literal listener, where there is one table per QUD which depends on # of alternatives
        func_tensor_dim = {
            "World": self.language.idx_to_world,
            "Alternative": lambda x: self.language.idx_to_alt(qud_idx, x),
            "QUD": self.language.idx_to_qud,
            "Context": lambda x: self.language.context_to_string(self.language.idx_to_context(x)),
            "Utterance": self.language.idx_to_utterance
        }
        vals = [
            [func_tensor_dim[dim](i) for i in range(d)]
            for d, dim in zip(tensor.size(), tensor_dims)
        ]
        rows = [list(val) + [prob.item()] for val, prob in zip(product(*vals), tensor.flatten())]
        cond_space = [d for d in tensor_dims if d not in output_space]
        comma = ", "
        p_string = f"{comma.join([s[0] for s in output_space])} | {comma.join([s[0] for s in cond_space])}"
        df = pd.DataFrame(rows, columns=tensor_dims + [p_string]).set_index(cond_space).pivot(columns=output_space)
        return df






    def test_literal_listener(self):
        assert(self.query_literal_listener("need visa?",
                                           "needs visa",
                                           "not US",
                                           torch.Tensor([1, 1, 1, 1]).bool()) == 1/3)   # Under-informative answer in weak context
        assert(self.query_literal_listener("need visa?",
                                           "needs visa",
                                           "not US",
                                           torch.Tensor([1, 0, 1, 0]).bool()) == 1)     # Under-informative answer in strong context
        assert(self.query_literal_listener("free drink?",
                                           "free drink",
                                           "green card",
                                           torch.Tensor([1, 1, 1, 1]).bool()) == 1)     # Exhaustive answer in weak context
        assert(self.query_literal_listener("free drink?",
                                           "no free drink",
                                           "green card",
                                           torch.Tensor([1, 1, 1, 1]).bool()) == 0)     # Exhaustive answer in weak context
        assert(self.query_literal_listener("free drink?",
                                           "no free drink",
                                           "US",
                                           torch.Tensor([1, 0, 1, 1]).bool()) == 1)     # Question is trivial, answer is true
        assert(self.query_literal_listener("free drink?",
                                           "no free drink",
                                           "green card",
                                           torch.Tensor([1, 0, 1, 1]).bool()).isnan())     # Answer contradicts context, question is trivial
        assert(self.query_literal_listener("free drink?",
                                           "no free drink",
                                           "green card",
                                           torch.Tensor([1, 0, 1, 1]).bool()).isnan())     # Answer contradicts context, answer is exhaustive




















