"""Print the reference stack's active transitive versions, never local paths."""
import importlib.metadata as metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
roots=['numpy','scipy','scikit-learn','requests','torch','transformers','tokenizers','safetensors','huggingface-hub','gguf','pytest','setuptools']
pending=[(name,set()) for name in roots];seen={};versions={}
while pending:
 name,extras=pending.pop();key=canonicalize_name(name)
 if key in seen and extras.issubset(seen[key]):continue
 seen.setdefault(key,set()).update(extras)
 dist=metadata.distribution(name);versions[key]=dist.version
 for raw in dist.requires or []:
  req=Requirement(raw)
  if req.marker and not any(req.marker.evaluate({'extra':extra}) for extra in (extras or {''})):continue
  pending.append((req.name,set(req.extras)))
print('# Reference Linux x86_64 Python3.10 stack; platform-specific, not a universal lock.')
print('# Install with the reference Python/platform. Other platforms use pyproject pins and record drift.')
for name,version in sorted(versions.items()):print(name+'=='+version)
