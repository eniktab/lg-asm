from pathlib import Path
from typing import Optional, List, Any, Tuple
import torch.nn as nn
from src.utils.HyenaBackend import HyenaBackend

def build_hyena_backend(
    model_dir: Path,
    model_name: str = "hyenadna-large-1m-seqlen-hf",
) -> HyenaBackend:
    backend = HyenaBackend(
        model_name=model_name,
        model_dir=str(model_dir),
        pooling="none",
        normalize=False,
        offline=True,
        prefer_cuda=True,
    )
    return backend

class HyenaLayerController:
    """
    Controls gradual unfreezing of Hyena backbone layers.

    Assumes that `model` is the Hyena *backbone* (without the classifier head).
    If you pass a full model (backbone + head), the head parameters will also
    be affected unless you manage them separately.

    Behavior:

    - If freeze_backbone is False:
        * All model parameters are trainable from epoch 0.
        * model.train() is called.

    - If freeze_backbone is True:
        * epochs < start_epoch:
            - All backbone parameters are frozen.
            - Backbone is put in eval() mode (no BN/Dropout updates).
        * epochs in [start_epoch, end_epoch]:
            - Linearly increase the number of unfrozen *top* layers
              up to max_unfrozen_layers.
        * epochs > end_epoch:
            - Keep max_unfrozen_layers unfrozen.

    Implementation details:

    - We try a list of common attribute paths first
      ("encoder.layers", "backbone.layers", "backbone.blocks", "layers",
       "blocks", "h") and use the first one that looks like a sequence of
      transformer blocks.

    - If that fails, we auto-detect a suitable ModuleList/Sequential by
      scanning named_modules(), scoring candidates by name and length.

    - When partially frozen:
        * Entire backbone is put in eval() mode and frozen.
        * The last N layers (N = n_unfrozen) are switched back to train()
          and have requires_grad=True, so only they update weights and
          BatchNorm / Dropout behave in training mode there.

    NOTE:
        To keep classifier heads fully trainable, either:
        (a) pass only the backbone into this controller, or
        (b) re-enable head modules' train() / requires_grad=True outside
            this controller after calling apply().
    """

    def __init__(
        self,
        model: nn.Module,
        freeze_backbone: bool,
        start_epoch: int,
        end_epoch: int,
        max_unfrozen_layers: int,
    ) -> None:
        self.model = model
        self.freeze_backbone = freeze_backbone
        self.start_epoch = start_epoch
        # Ensure end_epoch is not earlier than start_epoch
        self.end_epoch = max(start_epoch, end_epoch)
        # Negative values make no sense; clamp to >= 0
        self.max_unfrozen_layers = max(0, max_unfrozen_layers)

        self.layers: Optional[List[nn.Module]] = None

        # ------------------------------------------------------------
        # 1) Try explicit attribute paths first (most robust if they exist)
        # ------------------------------------------------------------
        candidates = (
            "encoder.layers",
            "backbone.layers",
            "backbone.blocks",
            "layers",
            "blocks",
            "h",
        )

        for attr_path in candidates:
            mod: Any = model
            ok = True
            for sub in attr_path.split("."):
                if not hasattr(mod, sub):
                    ok = False
                    break
                mod = getattr(mod, sub)
            if not ok:
                continue

            if isinstance(mod, (nn.ModuleList, nn.Sequential, list, tuple)):
                self.layers = list(mod)
                print(
                    f"[INFO] HyenaLayerController using explicit path '{attr_path}' "
                    f"with {len(self.layers)} layers."
                )
                break

        # ------------------------------------------------------------
        # 2) Fallback: auto-detect a good container of layers
        # ------------------------------------------------------------
        if self.layers is None:
            best_candidate: Optional[Tuple[str, nn.Module, int, float]] = None
            best_score = -1.0

            for name, module in self.model.named_modules():
                if isinstance(module, (nn.ModuleList, nn.Sequential)):
                    length = len(module)
                    if length < 2:
                        continue

                    lname = name.lower()
                    score = 0.0

                    # Prefer names that look like layer / block / stage containers
                    if any(key in lname for key in ("layers", "layer", "blocks", "block", "stage", "h")):
                        score += 10.0
                    # Prefer encoder/backbone/transformer-style containers
                    if any(key in lname for key in ("encoder", "backbone", "transformer")):
                        score += 5.0

                    # Slight bonus for more layers (capped)
                    score += min(float(length), 100.0)

                    if score > best_score:
                        best_score = score
                        best_candidate = (name, module, length, score)

            if best_candidate is not None:
                name, module, length, score = best_candidate
                self.layers = list(module)
                print(
                    f"[INFO] HyenaLayerController auto-detected layer container '{name}' "
                    f"with {length} layers (score={score:.1f})."
                )

        if self.layers is None:
            print(
                "[WARN] HyenaLayerController could not find a transformer-layer container; "
                "will treat the entire backbone as a single unit for freezing/unfreezing."
            )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def apply(self, epoch: int) -> None:
        """
        Apply the gradual unfreezing policy for a given epoch.

        This should typically be called once per epoch before running the
        training loop for that epoch. If your training code calls model.train()
        / model.eval() inside the epoch loop, call `apply()` *after* setting
        the global mode to ensure the partial train/eval flags on layers
        are not overwritten.
        """
        # --------------------------------------------------------
        # Case 0: no freezing strategy requested; ensure everything trains
        # --------------------------------------------------------
        if not self.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.train()
            return

        # --------------------------------------------------------
        # Compute how many layers should be unfrozen this epoch
        # --------------------------------------------------------
        if epoch < self.start_epoch:
            n_unfrozen = 0
        elif epoch >= self.end_epoch:
            n_unfrozen = self.max_unfrozen_layers
        else:
            # Linear schedule from start_epoch to end_epoch (inclusive)
            total_steps = max(1, self.end_epoch - self.start_epoch + 1)
            progress = (epoch - self.start_epoch + 1) / total_steps
            n_unfrozen = int(round(progress * self.max_unfrozen_layers))

        # Handle edge cases and clamp to valid range if we know the layers
        if self.layers is not None:
            n_unfrozen = max(0, min(n_unfrozen, len(self.layers)))

        # --------------------------------------------------------
        # If we do not know layers or the user requested 0 layers,
        # treat the whole backbone as a single block.
        # --------------------------------------------------------
        if self.layers is None or self.max_unfrozen_layers <= 0:
            if n_unfrozen <= 0:
                # Entire backbone frozen, eval mode
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()
                print(
                    f"[INFO] Epoch {epoch}: backbone fully frozen "
                    f"(no layer container detected or max_unfrozen_layers <= 0)."
                )
            else:
                # Gradual schedule degenerates to "freeze until some epoch,
                # then unfreeze everything".
                for p in self.model.parameters():
                    p.requires_grad = True
                self.model.train()
                print(
                    f"[INFO] Epoch {epoch}: backbone fully unfrozen "
                    f"(no layer container detected; treating as single block)."
                )
            return

        # --------------------------------------------------------
        # Normal case: we have a list of backbone layers.
        # Strategy:
        #   1. Put the entire backbone in eval() and freeze all params.
        #   2. Unfreeze last `n_unfrozen` layers and set them to train().
        # --------------------------------------------------------
        # 1) Base state: everything frozen, eval mode to avoid BN drift
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # 2) Unfreeze specific layers
        if n_unfrozen > 0:
            # Unfreeze last n_unfrozen layers (top of the stack)
            for layer in self.layers[-n_unfrozen:]:
                layer.train()  # enable Dropout/BN updates here
                for p in layer.parameters():
                    p.requires_grad = True

        # Note: if `model` is backbone-only, classifier heads are unaffected.
        # If `model` includes the head, remember to re-enable the head
        # explicitly in your training loop if needed.

        print(
            f"[INFO] Epoch {epoch}: unfreezing last {n_unfrozen} Hyena layers "
            f"(max_unfrozen_layers={self.max_unfrozen_layers}, "
            f"total_layers={len(self.layers)})."
        )
