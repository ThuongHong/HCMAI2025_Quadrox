"""Rerank options configuration and validation."""

from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RerankOptions:
    """Configuration options for multi-stage reranking pipeline."""

    # Master switches
    enable: bool = True
    mode: Literal["auto", "custom"] = "custom"

    # Sub-switches (SuperGlobal only)
    use_sg: bool = True

    # SuperGlobal parameters (tuned defaults for stability)
    sg_top_m: int = 200
    sg_qexp_k: int = 5
    sg_img_knn: int = 4
    # New parameters per SuperGlobal
    sg_alpha: float = 0.85
    sg_beta: float = 2.0
    sg_p_query: float = 80.0  # GeM power for query-side (~max but less sharp)
    # Backward-compat for legacy name
    sg_gem_p: float = 3.0
    w_sg: float = 1.0

    # Final output
    final_top_k: Optional[int] = 100

    # Cache and fallback controls
    cache_enabled: bool = True
    fallback_enabled: bool = True

    def __post_init__(self):
        """Validate and clamp parameters after initialization."""
        self._validate_and_clamp()

    def _validate_and_clamp(self):
        """Validate and clamp parameters to safe ranges."""
        # Clamp to safe ranges
        self.sg_top_m = max(1, min(10000, self.sg_top_m))
        self.sg_qexp_k = max(1, min(100, self.sg_qexp_k))
        # Allow img_knn=0 to disable DB-refine
        self.sg_img_knn = max(0, min(100, self.sg_img_knn))
        self.sg_alpha = float(min(max(self.sg_alpha, 0.0), 1.0))
        self.sg_beta = float(max(0.0, min(5.0, self.sg_beta)))
        self.sg_p_query = float(max(1.0, min(1000.0, self.sg_p_query)))
        self.sg_gem_p = max(0.1, min(1000.0, self.sg_gem_p))
        self.w_sg = max(0.0, min(5.0, self.w_sg))

        # Backward-compat: if sg_p_query not explicitly set (default) but legacy provided, map it
        # Heuristic: if user set sg_gem_p away from default 3.0 and didn't override sg_p_query, use sg_gem_p
        try:
            if 'sg_p_query' not in self.__dict__ or self.__dict__['sg_p_query'] == 100.0:
                # If legacy value seems customized, map it
                if self.sg_gem_p != 3.0:
                    self.sg_p_query = float(self.sg_gem_p)
        except Exception:
            pass

        # Handle final_top_k: None or 0 means no limit, positive means limit
        if self.final_top_k is not None and self.final_top_k > 0:
            self.final_top_k = max(1, min(1000, self.final_top_k))

        # Ensure final_top_k <= sg_top_m when positive
        if self.final_top_k is not None and self.final_top_k > 0 and self.final_top_k > self.sg_top_m:
            logger.warning(
                f"Clamping final_top_k from {self.final_top_k} to {self.sg_top_m}")
            self.final_top_k = self.sg_top_m

    @classmethod
    def from_request_and_config(
        cls,
        request_params: Dict[str, Any],
        config_defaults: Dict[str, Any]
    ) -> "RerankOptions":
        """
        Create RerankOptions from request parameters and config defaults.
        Precedence: Request params > ENV > config defaults
        """
        import os

        def get_value(key: str, default: Any = None, param_type: type = str):
            """Get value with precedence: request > env > config > default"""
            # Check request params first
            if key in request_params and request_params[key] is not None:
                try:
                    if param_type == bool:
                        return bool(request_params[key]) if isinstance(request_params[key], bool) else str(request_params[key]).lower() in ('1', 'true', 'on', 'yes')
                    elif param_type == int:
                        return int(request_params[key])
                    elif param_type == float:
                        return float(request_params[key])
                    else:
                        return str(request_params[key])
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid type for param {key}: {request_params[key]}, using default")

            # Check environment variables
            env_key = f"RERANK_{key.upper()}"
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    if param_type == bool:
                        return env_value.lower() in ('1', 'true', 'on', 'yes')
                    elif param_type == int:
                        return int(env_value)
                    elif param_type == float:
                        return float(env_value)
                    else:
                        return env_value
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid env value for {env_key}: {env_value}")

            # Check config defaults
            config_key = f"RERANK_{key.upper()}"
            if config_key in config_defaults:
                return config_defaults[config_key]

            return default

        # Build options with precedence
        options = cls(
            enable=get_value("enable", True, bool),
            mode=get_value("mode", "custom", str),

            use_sg=get_value("enable_superglobal", True, bool),

            sg_top_m=get_value("sg_top_m", 200, int),
            sg_qexp_k=get_value("sg_qexp_k", 5, int),
            sg_img_knn=get_value("sg_img_knn", 4, int),
            sg_alpha=get_value("sg_alpha", 0.85, float),
            sg_beta=get_value("sg_beta", 2.0, float),
            # Prefer sg_p_query, fall back to legacy sg_gem_p
            sg_p_query=get_value("sg_p_query", None, float) if get_value("sg_p_query", None, float) is not None else get_value("sg_gem_p", 80.0, float),
            sg_gem_p=get_value("sg_gem_p", 3.0, float),
            w_sg=get_value("w_sg", 1.0, float),

            final_top_k=get_value("final_top_k", 100, int),

            cache_enabled=get_value("cache_enabled", True, bool),
            fallback_enabled=get_value("fallback_enabled", True, bool),
        )

        # Validate at least one method is enabled in custom mode
        if options.enable and options.mode == "custom":
            if not options.use_sg:
                logger.warning("SuperGlobal disabled in custom mode, enabling it")
                options.use_sg = True

        return options

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "enable": self.enable,
            "mode": self.mode,
            "use_sg": self.use_sg,
            "sg_top_m": self.sg_top_m,
            "sg_qexp_k": self.sg_qexp_k,
            "sg_img_knn": self.sg_img_knn,
            "sg_alpha": self.sg_alpha,
            "sg_beta": self.sg_beta,
            "sg_p_query": self.sg_p_query,
            "final_top_k": self.final_top_k,
            "cache_enabled": self.cache_enabled,
            "fallback_enabled": self.fallback_enabled,
            "weights": {"w_sg": self.w_sg},
        }
