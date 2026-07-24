"""Environment sanity checks.

These are not tests of the model - they exist so that "it doesn't work" can be
localised to the environment or the code, never both at once. Run these first
whenever something is behaving strangely, and run them on Colab too.
"""

import torch

import tinygpt


def test_package_is_installed():
    """Imports resolve through the installed package, not a relative path.

    With the src/ layout this only passes after `pip install -e .`, which is the
    point: it fails the same way locally as it would on a fresh Colab VM.
    """
    assert tinygpt.__version__ == "0.1.0"


def test_torch_available():
    assert torch.__version__


def test_autograd_computes_a_known_gradient():
    """d/dx of x^2 at x=3 is 6. If this fails, nothing downstream is trustworthy."""
    x = torch.tensor(3.0, requires_grad=True)
    y = x**2
    y.backward()
    assert x.grad.item() == 6.0


def test_device_selection_never_hardcodes_cuda():
    """Mirrors the runtime fallback: config may ask for cuda, CPU must still work.

    See docs/adr/0002 - the laptop has no GPU and Colab does, so the same code
    has to run in both places.
    """
    requested = "cuda"
    device = requested if torch.cuda.is_available() else "cpu"
    torch.zeros(2, 2, device=device)  # must not raise in either environment
