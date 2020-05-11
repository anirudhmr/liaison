import os
import shutil
from pathlib import Path

from liaison.daper.dataset_constants import DATASET_PATH, LENGTH_MAP
from liaison.daper.milp.dataset import AUXILIARY_MILP, MILP

# DATASET_PREFIX = '/data/nms/tfp/'
DATASET_PREFIX = '/home/arc/vol/mnt/nms/tfp/'

for dtype in ['train', 'valid', 'test']:
  src = f'{DATASET_PREFIX}/datasets/milp/indset/size-100/{dtype}'
  dst = f'{DATASET_PREFIX}/datasets/milp/indset/size-100-cleaned/{dtype}'
  Path(dst).mkdir(parents=True, exist_ok=True)

  for fname in os.listdir(f'{DATASET_PREFIX}/dataset_infos/milp-indset-100/aux_info/{dtype}'):
    shutil.copy(f'{src}/{fname}', f'{dst}/{fname}')

  # collate together.
  for i, fname in enumerate(list(sorted(os.listdir(dst), key=lambda k: int(k[:-4])))):
    if i != int(fname[:-4]):
      shutil.move(f'{dst}/{fname}', f'{dst}/{i}.pkl')
