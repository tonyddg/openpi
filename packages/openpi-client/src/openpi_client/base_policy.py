import abc
from typing import Dict, List


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def infer_batch(self, obs_list: List[Dict]) -> List[Dict]:  # noqa: UP006
        """Infer actions from a batch of observations.

        Default implementation falls back to per-sample infer() calls.
        Subclasses that support native batched inference should override this.
        """
        return [self.infer(obs) for obs in obs_list]

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
