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
    # CIDEr weights n-grams by document frequency across the batch, so a
    # single-image corpus gives every n-gram the same IDF (score always 0.0).
    # Use a multi-image corpus so the metric has something to discriminate on.
    references = {
        "img1": ["a dog runs on the beach", "a brown dog runs along the shore"],
        "img2": ["a cat sleeps on the couch", "a lazy cat naps on the sofa"],
        "img3": ["a bird flies over the trees", "a small bird soars above the forest"],
    }
    close_hypotheses = {
        "img1": "a dog runs on the beach",
        "img2": "a cat sleeps on the couch",
        "img3": "a bird flies over the trees",
    }
    far_hypotheses = {
        "img1": "a cat sleeps on a couch",
        "img2": "a cat sleeps on the couch",
        "img3": "a bird flies over the trees",
    }

    close_score, _ = compute_cider(references, close_hypotheses)
    far_score, _ = compute_cider(references, far_hypotheses)
    assert close_score > far_score
