"""Behaviour tests for the BPE tokenizer, built one function at a time."""

from tinygpt.tokenizer import BPETokenizer, get_stats, merge


def test_get_stats_counts_adjacent_pairs():
    # (1,2) appears twice; (2,1) and (2,3) once each.
    assert get_stats([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}


def test_get_stats_empty_and_single_have_no_pairs():
    # Fewer than two ids means no adjacent pair exists - must not raise.
    assert get_stats([]) == {}
    assert get_stats([7]) == {}


def test_get_stats_does_not_mutate_input():
    ids = [5, 5, 5]
    get_stats(ids)
    assert ids == [5, 5, 5]


def test_merge_replaces_every_occurrence():
    assert merge([1, 2, 1, 2, 3], (1, 2), 256) == [256, 256, 3]


def test_merge_keeps_the_unmatched_tail():
    # Regression: an off-by-one loop bound used to drop the trailing element.
    assert merge([1, 2, 3], (1, 2), 256) == [256, 3]
    assert merge([5, 6], (7, 8), 256) == [5, 6]


def test_merge_is_non_overlapping_left_to_right():
    # Three identical ids yield one merge plus a leftover, not an overlap.
    assert merge([1, 1, 1], (1, 1), 256) == [256, 1]


def test_merge_handles_empty_and_single():
    assert merge([9], (1, 2), 256) == [9]
    assert merge([], (1, 2), 256) == []


def test_merge_does_not_mutate_input():
    ids = [1, 2, 3]
    merge(ids, (1, 2), 256)
    assert ids == [1, 2, 3]


def test_train_learns_hierarchical_merges():
    # "ababab" -> bytes [97,98,97,98,97,98]. The most frequent pair is (97,98),
    # and later merges compose earlier ones: (256,256) then (257,256).
    tok = BPETokenizer()
    tok.train("ababab", 259)
    assert tok.merges == {(97, 98): 256, (256, 256): 257, (257, 256): 258}


def test_train_mints_ids_from_256_upward():
    # Byte-level start means new ids never collide with the 0..255 byte range.
    tok = BPETokenizer()
    tok.train("aaaa", 258)
    assert min(tok.merges.values()) == 256
    assert sorted(tok.merges.values()) == [256, 257]


def test_fresh_tokenizer_has_no_merges():
    assert BPETokenizer().merges == {}


def test_decode_expands_merged_ids_to_text():
    tok = BPETokenizer()
    tok.train("ababab", 259)
    assert tok.decode([258]) == "ababab"  # a single merged id expands fully
    assert tok.decode([256, 256]) == "abab"


def test_decode_of_raw_bytes_needs_no_merges():
    # With no training, ids are just byte values; 104,105 -> "hi".
    assert BPETokenizer().decode([104, 105]) == "hi"


def test_encode_applies_learned_merges():
    tok = BPETokenizer()
    tok.train("ababab", 259)
    # "abab" -> bytes [97,98,97,98] -> (97,98) merges to 256 twice -> [256,256]
    # -> (256,256) merges to 257 -> [257].
    assert tok.encode("abab") == [257]


def test_encode_decode_round_trips():
    # The defining property of a tokenizer: decode(encode(text)) == text,
    # even for text (and unicode) never seen during training.
    tok = BPETokenizer()
    tok.train("the cat sat on the mat, the cat ran.", 320)
    for text in ["the cat", "hello world", "a", "", "unseen ééé"]:
        assert tok.decode(tok.encode(text)) == text


def test_encode_handles_empty_and_single_char():
    tok = BPETokenizer()
    tok.train("aaaa", 258)
    assert tok.encode("") == []
    assert tok.decode(tok.encode("a")) == "a"
