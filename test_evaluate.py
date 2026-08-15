from evaluate import compute_bleu, compute_cider


def test_bleu_is_high_for_identical_caption():
    references = {"img1": ["a dog runs on the beach"]}
    hypotheses = {"img1": "a dog runs on the beach"}
    scores = compute_bleu(references, hypotheses)
    assert scores["bleu1"] > 95
    assert scores["bleu4"] > 90


def test_bleu_is_low_for_unrelated_caption():
    references = {"img1": ["a dog runs on the beach"]}
    hypotheses = {"img1": "completely unrelated text about cars"}
    scores = compute_bleu(references, hypotheses)
    assert scores["bleu1"] < 30


def test_cider_is_higher_for_closer_match():
    references = {
        "img1": ["a dog runs on the beach", "a brown dog runs along the shore"],
    }
    close_hyp = {"img1": "a dog runs on the beach"}
    far_hyp = {"img1": "a cat sleeps on a couch"}

    close_score, _ = compute_cider(references, close_hyp)
    far_score, _ = compute_cider(references, far_hyp)
    assert close_score > far_score
