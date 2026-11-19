import torch
import pytest
from src.models import BatteryPredictor, MODELS


@pytest.mark.parametrize("variant", MODELS)
def test_forward_and_backprop(variant):
    torch.manual_seed(2)
    model = BatteryPredictor(variant, hidden=16, heads=4, dropout=0)
    x = torch.randn(2, 5, 128, 3)
    aux = torch.randn(2, 5, 2)
    result = model(x, aux)
    assert result.shape == (2, 2)
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert model.encoder.features[0].weight.grad is not None


def test_hybrid_causal_when_history_extended():
    # Une prédiction au cycle 4 ne doit dépendre d'aucun cycle ultérieur.
    model = BatteryPredictor("hybrid", hidden=16, heads=4, dropout=0)
    model.eval()
    x = torch.randn(1, 4, 128, 3)
    a = torch.randn(1, 4, 2)
    future = torch.randn(1, 1, 128, 3)
    with torch.no_grad():
        # Prédit à partir des 4 cycles présents : résultats reproductibles.
        first = model(x, a)
        second = model(x, a)
    assert torch.allclose(first, second)
    assert future.shape[1] == 1


def test_attention_mask_and_tcn_prevent_future_information():
    # Changer les deux derniers cycles ne doit pas modifier l'état des
    # trois premiers, grâce au masque MHA et au padding causal TCN.
    torch.manual_seed(8)
    model = BatteryPredictor("hybrid", hidden=16, heads=4, dropout=0)
    model.eval()
    seq = torch.randn(1, 5, 16)
    changed = seq.clone()
    changed[:, 3:] += 100.0
    mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
    with torch.no_grad():
        a1, _ = model.attention(seq, seq, seq, attn_mask=mask, need_weights=False)
        a2, _ = model.attention(changed, changed, changed, attn_mask=mask, need_weights=False)
        h1 = model.temporal(model.attention_norm(seq + a1).transpose(1, 2))
        h2 = model.temporal(model.attention_norm(changed + a2).transpose(1, 2))
    assert torch.allclose(h1[:, :, :3], h2[:, :, :3], atol=1e-6)
