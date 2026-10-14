"""Compare this session's detrended re-tune cache with the one left by the interrupted run."""
import pickle
from pathlib import Path
H = Path(__file__).parent
a = pickle.loads((H / "v2_detrended_cache.pkl").read_bytes())
b = pickle.loads((H / "detrended_cache.pkl").read_bytes())
diff = sum(1 for x, y in zip(a, b) if x != y)
print("configs", len(a), len(b), "configs with any difference:", diff)
