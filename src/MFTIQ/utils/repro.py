import subprocess
from MFTIQ import repo_path
import logging
logger = logging.getLogger(__name__)


def code_export(path):
    env_path = str(repo_path)  # the main repo directory
    cmd = ["rsync", "-avm",
           # "--dry-run"
           ]

    excluded = ['.git', '__pycache__', 'pytracking_old', '.ipynb_checkpoints',
                'logs',
                # RAFT stuff
                'checkpoints', 'flowou_evals', 'traintxt', 'demo-frames',
                'checkpoint_debug', 'logs_debug', 'runs', 'train_files_lists',
                'experiments', 'export', 'demo_in', 'demo_out']
    for x in excluded:
        cmd.append(f"--exclude='{x}'")
    path.mkdir(parents=True, exist_ok=True)
    cmd += ["--include='*/'", "--include='*.py'", "--exclude='*'", "./",
            str(path.expanduser().resolve())]
    subprocess.check_call(' '.join(cmd), cwd=env_path, shell=True,
                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    logger.info(f'MFT repo backed up at {path}')

