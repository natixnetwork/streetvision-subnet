from dataclasses import dataclass, field
import random
from typing import Iterable, List, Optional, Set

import bittensor as bt
import numpy as np

def check_uid_availability(metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int) -> bool:
    if not metagraph.axons[uid].is_serving:
        return False
    if metagraph.validator_permit[uid] and metagraph.S[uid] > vpermit_tao_limit:
        return False
    return True

@dataclass
class UIDDeck:
    rng: random.Random = field(default_factory=random.Random)
    deck: List[int] = field(default_factory=list)
    idx: int = 0

    def _build_eligible(self, metagraph, vpermit_tao_limit: int, exclude: Optional[Set[int]] = None) -> List[int]:
        exclude = exclude or set()
        out: List[int] = []
        for uid in range(metagraph.n.item()):
            if uid in exclude:
                continue
            if check_uid_availability(metagraph, uid, vpermit_tao_limit):
                out.append(uid)
        return out

    def refill(self, metagraph, vpermit_tao_limit: int, exclude: Optional[Iterable[int]] = None) -> None:
        eligible = self._build_eligible(metagraph, vpermit_tao_limit, set(exclude or []))
        self.deck = eligible
        self.rng.shuffle(self.deck)
        self.idx = 0

    def next_order(
        self,
        metagraph,
        vpermit_tao_limit: int,
        exclude: Optional[Iterable[int]] = None,
    ) -> List[int]:
        # Ensure deck exists and is not exhausted
        if not self.deck or self.idx >= len(self.deck):
            self.refill(metagraph, vpermit_tao_limit, exclude)

        # Return remaining order for this cycle
        return self.deck[self.idx:].copy()

    # organic selection will “scan” candidates and then advance by how many it consumed from the deck.
    def advance(self, n: int) -> None:
        self.idx = min(self.idx + n, len(self.deck))

    def next_k(
        self,
        k: int,
        metagraph,
        vpermit_tao_limit: int,
        exclude: Optional[Iterable[int]] = None,
    ) -> np.ndarray:
        # Refill if empty or exhausted
        if not self.deck or self.idx >= len(self.deck):
            self.refill(metagraph, vpermit_tao_limit, exclude)

        if not self.deck:
            return np.array([], dtype=np.int64)

        out: List[int] = []
        exclude_set = set(exclude or [])

        # Fill up to k, reshuffling as needed
        while len(out) < k:
            if self.idx >= len(self.deck):
                self.refill(metagraph, vpermit_tao_limit, exclude_set)
                if not self.deck:
                    break

            uid = self.deck[self.idx]
            self.idx += 1

            # re-check availability at draw time to handle churn
            if uid in exclude_set:
                continue
            if not check_uid_availability(metagraph, uid, vpermit_tao_limit):
                continue

            out.append(uid)

        return np.array(out, dtype=np.int64)