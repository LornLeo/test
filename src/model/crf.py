"""Linear-chain CRF and Viterbi with explicit BIO start/transition constraints."""

import torch
from torch import nn


class BIOCRF(nn.Module):
    def __init__(self, tags: list[str], learn_transitions: bool = True):
        super().__init__()
        size = len(tags)
        self.transitions = nn.Parameter(torch.zeros(size, size), requires_grad=learn_transitions)
        self.start = nn.Parameter(torch.zeros(size), requires_grad=learn_transitions)
        self.end = nn.Parameter(torch.zeros(size), requires_grad=learn_transitions)
        allowed = torch.ones(size, size, dtype=torch.bool)
        start_allowed = torch.ones(size, dtype=torch.bool)
        for current, tag in enumerate(tags):
            if tag.startswith("I-"):
                start_allowed[current] = False
                allowed[:, current] = torch.tensor(
                    [previous in ("B-" + tag[2:], "I-" + tag[2:]) for previous in tags]
                )
        self.register_buffer("allowed", allowed)
        self.register_buffer("start_allowed", start_allowed)

    def scores(self):
        # A finite sentinel avoids undefined gradients for impossible intermediate
        # states; it remains far below any practical sequence score.
        return (
            self.transitions.masked_fill(~self.allowed, -10000.0),
            self.start.masked_fill(~self.start_allowed, -10000.0),
            self.end,
        )

    def neg_log_likelihood(self, emissions, tags, mask):
        if not bool(mask[:, 0].all()):
            raise ValueError("CRF sequences must contain at least one token")
        transitions, start, end = self.scores()
        safe_tags = tags.masked_fill(~mask, 0)
        if not bool(self.start_allowed[safe_tags[:, 0]].all()):
            raise ValueError("Illegal BIO start in Gold")
        if not bool((self.allowed[safe_tags[:, :-1], safe_tags[:, 1:]] | ~mask[:, 1:]).all()):
            raise ValueError("Illegal BIO transition in Gold")
        alpha = start + emissions[:, 0]
        score = start[safe_tags[:, 0]] + emissions[:, 0].gather(1, safe_tags[:, :1]).squeeze(1)
        for time in range(1, emissions.shape[1]):
            proposed = (
                torch.logsumexp(alpha.unsqueeze(2) + transitions.unsqueeze(0), dim=1)
                + emissions[:, time]
            )
            alpha = torch.where(mask[:, time, None], proposed, alpha)
            previous, current = safe_tags[:, time - 1], safe_tags[:, time]
            increment = transitions[previous, current] + emissions[:, time].gather(
                1, current[:, None]
            ).squeeze(1)
            score += increment * mask[:, time]
        last = safe_tags.gather(1, (mask.sum(1) - 1)[:, None]).squeeze(1)
        score += end[last]
        return torch.logsumexp(alpha + end, dim=1) - score

    @torch.no_grad()
    def decode(self, emissions, mask):
        transitions, start, end = self.scores()
        scores = start + emissions[:, 0]
        pointers = []
        for time in range(1, emissions.shape[1]):
            proposed, pointer = (scores.unsqueeze(2) + transitions.unsqueeze(0)).max(dim=1)
            proposed += emissions[:, time]
            scores = torch.where(mask[:, time, None], proposed, scores)
            pointers.append(pointer)
        last_tags = (scores + end).argmax(dim=1).tolist()
        # Transfer once for Python backtracking instead of synchronizing the GPU
        # for every token/sequence pointer access.
        pointers = torch.stack(pointers).cpu() if pointers else []
        allowed = self.allowed.cpu()
        start_allowed = self.start_allowed.cpu()
        paths = []
        for row, length in enumerate(mask.sum(1).tolist()):
            path = [last_tags[row]]
            for time in range(length - 2, -1, -1):
                path.append(int(pointers[time][row, path[-1]]))
            path.reverse()
            if not bool(start_allowed[path[0]]) or any(
                not bool(allowed[a, b]) for a, b in zip(path, path[1:])
            ):
                raise RuntimeError("BIO decoder produced an illegal path")
            paths.append(path)
        return paths
