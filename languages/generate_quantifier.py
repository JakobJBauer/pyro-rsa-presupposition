
# n people
# for each person, there are 3 options
# 3^n possible worlds
# thus there are 2^(3^n) possible belief states
# This is wayyyyy too many. Let's cut this down...


# n people
# for each person, there are 3 options, but we don't care who is in which state, just how many
# n_us + n_gc + n_else = n
# The number of ways to write this sum is C(n+2, 2) (the triangular numbers).
# In general, if there are k mutually exclusive states,
# then the number of cells in the partition is C(n+k-2, k-1)

n_people = 3
idx_to_world = dict(enumerate([(i, j, n_people-i-j) for i in range(n_people+1) for j in range(n_people+1-i)]))
world_to_idx = {v: k for k, v in idx_to_world.items()}
n_worlds = len(idx_to_world)

# Now, there are C(n+2, 2) cells, meaning there is 2^C(n+2, 2) belief states.
# Let's say there are 5 people. This gives us 21 cells, and 2^21 ~= 2E6 belief states.
# This is still A LOT to do inference over

# So we can restrict the number of belief states as follows:
# Why should you be able to believe that the possible cells are {(6, 4, 0), (4, 6, 0)}, but not (5, 5, 0)?
# You should be able to believe "between m and n people have property p"

# Todo: Find a more efficient approach
bounds = [(i, j) for i in range(n_people+1) for j in range(i, n_people+1)]
subsets = []
for b1 in bounds:
    for b2 in bounds:
        for b3 in bounds:
            subset = []
            for v1 in range(b1[0], b1[1]+1):
                for v2 in range(b2[0], b2[1]+1):
                    v3 = n_people - (v1 + v2)
                    if v1 + v2 + v3 == n_people and v3 >= b3[0] and v3 <= b3[1]:
                        subset.append((v1, v2, v3))
            subsets.append(tuple(subset))
subsets = set(subsets)
subsets.remove(tuple())
idx_to_belief_state = dict(enumerate(subsets))


# The above represents a belief state as a list of tuples (sums) of properties
# We want to represent a world as boolean vector of length n_worlds
import torch

zeros = torch.Tensor([0]).expand(n_worlds)
def world_list_to_proposition(world_list):
    return zeros.clone().index_fill_(0, torch.tensor(world_list), 1)


# zeros = torch.Tensor([0]).expand(n_worlds)
belief_states_world_indices = [
    [world_to_idx[w] for w in bs]
    for i, bs in idx_to_belief_state.items()
]
belief_states = torch.stack(
    [
        world_list_to_proposition(bs)
        for bs in belief_states_world_indices
    ]
)


# QUD: The QUD is "How many people need a visa?" Property 2 indicates this value:
n_alternatives = n_people + 1
from collections import defaultdict
partitions = defaultdict(list)
for idx, w in idx_to_world.items():
    partitions[w[2]].append(idx)

qud = {
    "How many visas?": {
        str(i): world_list_to_proposition(partitions[i]).int().tolist()
        for i in range(n_alternatives)
    }
}


# Other QUD: For testing purposes we need "How many non-US citizens?" Property 0 indicates this value of US citizens:
partitions = defaultdict(list)
for idx, w in idx_to_world.items():
    partitions[w[0]].append(idx)

qud["How many non-US?"] = {
    str(n_alternatives - i - 1): world_list_to_proposition(partitions[i]).int().tolist()
    for i in range(n_alternatives)
}


# Utterances: Let's say {all, some, none} X {green card, not green card, US, not US}
quantifiers = ["all", "some", "none"]
properties = ["US", "not US", "green card", "not green card", "need visa", "not need visa"]

utterances = {}
for p in properties:
    indices = {
        "US": [0],
        "not US": [1, 2],
        "green card": [1],
        "not green card": [0, 2],
        "need visa": [2],
        "not need visa": [0, 1]
    }[p]
    for q in quantifiers:
        if q == "all":
            if "not" in p:
                continue
            proposition = world_list_to_proposition(
                [i_w for i_w, w in idx_to_world.items() if sum([w[i_p] for i_p in indices]) == n_people]
            )
        elif q == "some":
            proposition = world_list_to_proposition(
                [i_w for i_w, w in idx_to_world.items() if sum([w[i_p] for i_p in indices]) > 0]
            )
        else:  # if q == "none":
            if "not" in p:
                continue
            proposition = world_list_to_proposition(
                [i_w for i_w, w in idx_to_world.items() if sum([w[i_p] for i_p in indices]) == 0]
            )
        utterances[f"{q} {p}"] = proposition.int().tolist()

utterances["_"] = torch.tensor([1]).expand(n_worlds).tolist()

json_str = {
    "quds": qud,
    "utterances": utterances,
    "n_worlds": n_worlds,
    "worlds": idx_to_world,
    "contexts": belief_states.int().tolist()
}

import json
with open("quantifier.json", "w") as f:
    f.write(json.dumps(json_str))
