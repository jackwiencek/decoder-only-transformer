def get_stats(ids: list[int]) -> dict:
    count = {}
    for pair in zip(ids, ids[1:], strict=False):  # unequal lengths on purpose
        count[pair] = count.get(pair, 0) + 1
    return count


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    i = 0
    new_ids_list = []
    while i < len(ids):
        if i < (len(ids) - 1) and (ids[i], ids[i + 1]) == pair:
            new_ids_list.append(new_id)
            i += 2
        else:
            new_ids_list.append(ids[i])
            i += 1
    return new_ids_list


class BPETokenizer:
    def __init__(self):
        self.merges = {}

    def train(self, text: str, vocab_size: int) -> None:
        ids = list(text.encode("utf-8"))
        pair_id_dict = {}
        for new_id in range(256, vocab_size):
            if len(ids) < 2:
                break  # nothing left to merge; corpus too small for this vocab
            current_counts = get_stats(ids)
            most_freq_pair = max(current_counts, key=current_counts.get)
            pair_id_dict[most_freq_pair] = new_id
            ids = merge(ids, most_freq_pair, new_id)
        self.merges = pair_id_dict

    def decode(self, ids: list[int]) -> str:
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), new_id in self.merges.items():
            vocab[new_id] = vocab[p0] + vocab[p1]
        return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        # Start from raw bytes, then replay learned merges in the SAME order
        # train created them: at each step apply the pair with the lowest merge
        # id (earliest learned) that is currently present in the sequence.
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Rank present pairs by their merge id; pairs never learned get
            # infinity so they are never selected.
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing left in the sequence is mergeable
            ids = merge(ids, pair, self.merges[pair])
        return ids
