import torch
from torch import Tensor
from utils import _normalize

class Language:

    def __init__(self, language_file):
        """
        worlds = [w_0, ..., w_n]
        quds = {q_0: A_0 x W, ..., q_m: A_m x W}
        utterances = {}
        """
        self.language_specs = eval(open(language_file).read())
        self.n_worlds = self.language_specs["n_worlds"]
        self.worlds = torch.arange(0, self.n_worlds)  # 0: US | 1: GC | 2: non-US, non-GC, needs visa | 3: non-US, non-GC, doesn't need visa
        self._idx_to_world, self._world_to_idx = self.make_worlds()
        self.quds, self._idx_to_alt, self._alt_to_idx, self._idx_to_qud, self._qud_to_idx, self.qud_costs = self.make_quds()
        self.n_quds = len(self.quds)
        self.utterances, self._idx_to_utt, self._utt_to_idx, self.costs = self.make_utterances()
        self.n_utterances = len(self.utterances)
        self.n_contexts, self._idx_to_context, self._context_to_idx, self.contexts = self.make_contexts()

    def make_worlds(self):
        if type(self.language_specs["worlds"].values().__iter__().__next__()) == list:
            f = lambda x: tuple(x)
        else:
            f = lambda x: x
        idx_to_world = {int(i): f(w) for i, w in self.language_specs["worlds"].items()}
        world_to_idx = {f(w): i for i, w in idx_to_world.items()}
        return idx_to_world, world_to_idx

    def make_quds(self):
        qud_structure = self.language_specs["quds"]
        quds = {
            i: torch.stack([torch.Tensor(x).bool() for x in qud_structure[q].values()])
            for i, q in enumerate(qud_structure.keys())
        }
        idx_to_qud = {i: q for i, q in enumerate(qud_structure.keys())}
        qud_to_idx = {q: i for i, q in enumerate(qud_structure.keys())}
        idx_to_alt = {
            qud_to_idx[q]: {i: a for i, a in enumerate(qud_structure[q].keys())}
            for q in qud_structure.keys()
        }
        alt_to_idx = {
            qud_to_idx[q]: {a: i for i, a in enumerate(qud_structure[q].keys())}
            for q in qud_structure.keys()
        }
        qud_costs = torch.Tensor([self.cost(u) for u in qud_to_idx]).bool()
        return quds, idx_to_alt, alt_to_idx, idx_to_qud, qud_to_idx, qud_costs

    def idx_to_alt(self, qud, idx):
        if type(qud) is str:
            qud = self._qud_to_idx[qud]
        return self._idx_to_alt[qud][idx]

    def alt_to_idx(self, qud, alt):
        if type(qud) is str:
            qud = self._qud_to_idx[qud]
        return self._alt_to_idx[qud][alt]


    def make_utterances(self):
        utterance_structure = self.language_specs["utterances"]
        utterances = torch.stack([torch.Tensor(x).bool() for x in utterance_structure.values()])
        # utterance_structure = {
        #     "US": torch.Tensor([1, 0, 0, 0]).bool(),
        #     "not US": torch.Tensor([0, 1, 1, 1]).bool(),
        #     "green card": torch.Tensor([0, 1, 0, 0]).bool(),
        #     "not green card": torch.Tensor([1, 0, 1, 1]).bool(),
        #     "_": torch.Tensor([1, 1, 1, 1]).bool()
        # }
        # utterances = torch.stack(list(utterance_structure.values()))
        idx_to_utt = {i: a for i, a in enumerate(utterance_structure.keys())}
        utt_to_idx = {a: i for i, a in enumerate(utterance_structure.keys())}
        costs = torch.Tensor([self.cost(u) for u in utt_to_idx]).bool()
        return utterances, idx_to_utt, utt_to_idx, costs

    def n_alternatives(self, qud):
        return len(self.quds[qud])

    def make_contexts(self):
        if "contexts" in self.language_specs:
            n_contexts = len(self.language_specs["contexts"])
            idx_to_context = dict(enumerate([torch.IntTensor(c).bool() for c in self.language_specs["contexts"]]))
        else:
            n_contexts = 2**self.n_worlds - 1
            idx_to_context = {i: self.idx_to_context(i) for i in range(n_contexts)}
        context_to_idx = {c: i for i, c in idx_to_context.items()}
        contexts = torch.stack(list(idx_to_context.values()))
        return n_contexts, idx_to_context, context_to_idx, contexts

    def cost(self, utterance):
        if utterance == "_":
            return 0
        else:
            return 1

    def idx_to_context(self, i):
        if "contexts" in self.language_specs:
            return self._idx_to_context[i]
        else:
            mask = 2 ** torch.arange(self.n_worlds)
            return torch.IntTensor([i]).bitwise_and(mask).eq(0).bool()

    def context_to_idx(self, context):
        if "contexts" in self.language_specs:
            return self._context_to_idx[context]
        else:
            mask = 2 ** torch.arange(self.n_worlds)
            return sum(torch.bitwise_not(context.bool()).int() * mask).item()

    def context_distr_to_world_distr(self, context_distr):
        prod = self.contexts.expand(self.n_utterances, -1, -1) * context_distr.expand(self.n_worlds, -1, -1).transpose(0, 2)
        inferred_worlds_prior = torch.sum(_normalize(prod, dim=(1, 2)), dim=1)

    def context_to_string(self, c):
        if "context_to_name" in self.language_specs:
            return self.language_specs["context_to_name"][str(self.context_to_idx(c))]
        else:
            return "".join([str(x) for x in c.int().numpy()])

    def world_to_idx(self, w):
        return self._world_to_idx[w]

    def idx_to_world(self, i):
        return self._idx_to_world[i]

    def utterance_to_idx(self, u):
        return self._utt_to_idx[u]

    def idx_to_utterance(self, i):
        return self._idx_to_utt[i]

    def qud_to_idx(self, q):
        return self._qud_to_idx[q]

    def idx_to_qud(self, i):
        return self._idx_to_qud[i]

    def pretty_print_context(self, c):
        worlds = [
            self.idx_to_world(i)
            for i, b in enumerate(c)
            if b == 1
        ]
        return worlds
