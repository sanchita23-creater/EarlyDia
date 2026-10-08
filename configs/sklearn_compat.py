"""
EarlyDia - scikit-learn pickle compatibility helpers.

Why this exists (root cause of ``ModuleNotFoundError: No module named '_loss'``)
------------------------------------------------------------------------------
The production artifacts contain a fitted ``HistGradientBoostingClassifier``. Its loss
object is a Cython extension type whose ``__module__`` is the bare name ``'_loss'``
(not ``'sklearn._loss._loss'``), so the pickle stream stores the reference
``_loss.CyHalfBinomialLoss``. While unpickling, Python must ``import _loss`` as a
TOP-LEVEL module. That only works if something has already registered the extension
under the key ``sys.modules['_loss']``. Some interpreter/scikit-learn builds do this as a
side-effect of importing ``sklearn._loss._loss``; others (e.g. certain Windows builds)
do not, and the load fails with ``ModuleNotFoundError``.

The fix keeps the artifacts byte-for-byte unchanged and resolves the reference at load
time, in two complementary ways:

* ``load_artifact`` unpickles with a custom ``Unpickler`` whose ``find_class`` remaps the
  bare ``_loss`` module to ``sklearn._loss._loss``. It has no global side effects.
* ``register_loss_alias`` registers the same alias in ``sys.modules`` (idempotent) so
  utility scripts that still call plain ``pickle.load`` keep working.

Neither function retrains, re-fits, re-saves or otherwise modifies any model artifact.
"""
import importlib
import pickle
import sys
from pathlib import Path

#: scikit-learn version the artifacts were trained/verified with (see requirements.txt)
EXPECTED_SKLEARN_VERSION = "1.8.0"
_BARE_LOSS_MODULE = "_loss"
_REAL_LOSS_MODULE = "sklearn._loss._loss"


def register_loss_alias() -> None:
    """Idempotently expose ``sklearn._loss._loss`` under ``sys.modules['_loss']``."""
    if _BARE_LOSS_MODULE not in sys.modules:
        sys.modules[_BARE_LOSS_MODULE] = importlib.import_module(_REAL_LOSS_MODULE)


class _CompatUnpickler(pickle.Unpickler):
    """Resolves the legacy bare ``_loss`` module name to ``sklearn._loss._loss``."""

    def find_class(self, module, name):
        if module == _BARE_LOSS_MODULE:
            module = _REAL_LOSS_MODULE
        return super().find_class(module, name)


def load_artifact(path) -> dict:
    """Load a production pipeline artifact (read-only) with the compatibility remap.

    Raises ``RuntimeError`` with an actionable message when the installed scikit-learn is
    incompatible with the pickled model, instead of a bare low-level traceback.
    """
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            return _CompatUnpickler(fh).load()
    except (ModuleNotFoundError, AttributeError, ImportError, TypeError, ValueError) as exc:
        try:
            import sklearn
            installed = sklearn.__version__
        except Exception:  # pragma: no cover - sklearn missing entirely
            installed = "not installed"
        raise RuntimeError(
            f"Could not load model artifact '{path.name}' ({type(exc).__name__}: {exc}). "
            f"The artifacts were produced with scikit-learn {EXPECTED_SKLEARN_VERSION}; "
            f"installed: {installed}. Run `pip install -r requirements.txt` in the active "
            f"environment. The model files themselves must NOT be regenerated."
        ) from exc
