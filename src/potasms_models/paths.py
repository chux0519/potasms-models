from pathlib import Path
import os
DATA=Path(os.environ.get('POTASMS_WORK','work')).resolve()
CORPUS=DATA/'corpora/processed'
RUN=DATA/'run'
MODELS=DATA/'models'
TEACHER=MODELS/'bge'
