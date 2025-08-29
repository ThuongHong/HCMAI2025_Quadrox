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

    # Sub-switches
    use_sg: bool = True
    use_caption: bool = False
    use_llm: bool = False

    # SuperGlobal parameters
    sg_top_m: int = 500
    sg_qexp_k: int = 10
    sg_img_knn: int = 10
    sg_gem_p: float = 3.0
    w_sg: float = 1.0

    # Caption parameters
    cap_top_t: int = 20
    cap_model: str = "5CD-AI/Vintern-1B-v2"
    cap_max_tokens: int = 64
    cap_temp: float = 0.0
    w_cap: float = 0.8

    # LLM parameters
    llm_top_t: int = 5
    llm_model: str = "5CD-AI/Vintern-1B-v2"
    llm_timeout: int = 15
    w_llm: float = 1.2

    # Final output
    final_top_k: int = 100

    def __post_init__(self):
        """Validate and clamp parameters after initialization."""
        self._validate_and_clamp()

    def _validate_and_clamp(self):
        """Validate and clamp parameters to safe ranges."""
        # Clamp to safe ranges
        self.sg_top_m = max(1, min(10000, self.sg_top_m))
        self.sg_qexp_k = max(1, min(100, self.sg_qexp_k))
        self.sg_img_knn = max(1, min(100, self.sg_img_knn))
        self.sg_gem_p = max(0.1, min(10.0, self.sg_gem_p))
        self.w_sg = max(0.0, min(5.0, self.w_sg))

        self.cap_top_t = max(1, min(100, self.cap_top_t))
        self.cap_max_tokens = max(1, min(512, self.cap_max_tokens))
        self.cap_temp = max(0.0, min(2.0, self.cap_temp))
        self.w_cap = max(0.0, min(5.0, self.w_cap))

        self.llm_top_t = max(1, min(20, self.llm_top_t))
        self.llm_timeout = max(1, min(300, self.llm_timeout))
        self.w_llm = max(0.0, min(5.0, self.w_llm))

        self.final_top_k = max(1, min(1000, self.final_top_k))

        # Enforce ordering constraints: llm_top_t <= cap_top_t <= final_top_k <= sg_top_m
        if self.llm_top_t > self.cap_top_t:
            logger.warning(
                f"Clamping llm_top_t from {self.llm_top_t} to {self.cap_top_t}")
            self.llm_top_t = self.cap_top_t

        if self.cap_top_t > self.final_top_k:
            logger.warning(
                f"Clamping cap_top_t from {self.cap_top_t} to {self.final_top_k}")
            self.cap_top_t = self.final_top_k

        if self.final_top_k > self.sg_top_m:
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
            use_caption=get_value("enable_caption", False, bool),
            use_llm=get_value("enable_llm", False, bool),

            sg_top_m=get_value("sg_top_m", 500, int),
            sg_qexp_k=get_value("sg_qexp_k", 10, int),
            sg_img_knn=get_value("sg_img_knn", 10, int),
            sg_gem_p=get_value("sg_gem_p", 3.0, float),
            w_sg=get_value("w_sg", 1.0, float),

            cap_top_t=get_value("cap_top_t", 20, int),
            cap_model=get_value("cap_model", "5CD-AI/Vintern-1B-v2", str),
            cap_max_tokens=get_value("cap_max_tokens", 64, int),
            cap_temp=get_value("cap_temp", 0.0, float),
            w_cap=get_value("w_cap", 0.8, float),

            llm_top_t=get_value("llm_top_t", 5, int),
            llm_model=get_value("llm_model", "5CD-AI/Vintern-1B-v2", str),
            llm_timeout=get_value("llm_timeout", 15, int),
            w_llm=get_value("w_llm", 1.2, float),

            final_top_k=get_value("final_top_k", 100, int),
        )

        # Validate at least one method is enabled in custom mode
        if options.enable and options.mode == "custom":
            if not (options.use_sg or options.use_caption or options.use_llm):
                logger.warning(
                    "No rerank methods enabled in custom mode, falling back to SuperGlobal")
                options.use_sg = True

        return options

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "enable": self.enable,
            "mode": self.mode,
            "use_sg": self.use_sg,
            "use_caption": self.use_caption,
            "use_llm": self.use_llm,
            "sg_top_m": self.sg_top_m,
            "cap_top_t": self.cap_top_t,
            "llm_top_t": self.llm_top_t,
            "final_top_k": self.final_top_k,
            "weights": {
                "w_sg": self.w_sg,
                "w_cap": self.w_cap,
                "w_llm": self.w_llm
            }
        }
