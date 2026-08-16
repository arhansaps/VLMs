import torch
import torch.nn as nn

from model import VLM
from train import build_lr_lambda, build_param_groups, compute_loss, is_new_best


def test_is_new_best_when_no_prior_best():
    assert is_new_best(1.5, None) is True


def test_is_new_best_when_improved():
    assert is_new_best(1.0, 1.5) is True


def test_is_new_best_when_not_improved():
    assert is_new_best(2.0, 1.5) is False


def test_build_param_groups_separates_projection_and_gpt2():
    model = VLM()
    groups = build_param_groups(model, projection_lr=1e-4, gpt2_lr=1e-5)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 1e-5

    projection_param_ids = {id(p) for p in model.projection.parameters()}
    group0_ids = {id(p) for p in groups[0]["params"]}
    assert projection_param_ids == group0_ids


def test_lr_lambda_warms_up_linearly():
    lr_lambda = build_lr_lambda(warmup_steps=10, total_steps=100)
    assert lr_lambda(0) == 0.0
    assert lr_lambda(5) == 0.5
    assert lr_lambda(10) == 1.0


def test_lr_lambda_decays_after_warmup():
    lr_lambda = build_lr_lambda(warmup_steps=10, total_steps=100)
    assert lr_lambda(100) < 0.01
    assert lr_lambda(55) < lr_lambda(10)


def test_compute_loss_runs_for_single_visual_token():
    model = VLM(spatial_visual_tokens=False)
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 50257, (2, 10))
    attention_mask = torch.ones(2, 10)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    logits = model(images, input_ids, attention_mask)
    loss = compute_loss(logits, input_ids, attention_mask, criterion)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_compute_loss_runs_for_49_spatial_visual_tokens():
    # This is the case that broke: shift_logits/labels only lined up by
    # coincidence when there was exactly 1 visual token.
    model = VLM(spatial_visual_tokens=True)
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, 50257, (2, 10))
    attention_mask = torch.ones(2, 10)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    logits = model(images, input_ids, attention_mask)
    loss = compute_loss(logits, input_ids, attention_mask, criterion)

    assert loss.dim() == 0
    assert torch.isfinite(loss)
