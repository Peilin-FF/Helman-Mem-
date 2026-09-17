# kalman-mem: only the base and database environments are needed; the others are optional (their dependencies are not installed)
from .base_env import BaseEnvironment
from .db_env import DBEnvironment

try:
    from .coding_env import CodingEnvironment
except Exception:   # pragma: no cover
    CodingEnvironment = None
try:
    from .web_env import WebEnvironment
except Exception:   # pragma: no cover
    WebEnvironment = None
try:
    from .research_env import ResearchEnvironment
except Exception:   # pragma: no cover
    ResearchEnvironment = None
try:
    from .minecraft_env import MinecraftEnvironment
except Exception:   # pragma: no cover
    MinecraftEnvironment = None
try:
    from .world_env import WorldSimulationEnvironment
except Exception:   # pragma: no cover
    WorldSimulationEnvironment = None
