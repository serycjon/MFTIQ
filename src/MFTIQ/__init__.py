from pathlib import Path
from MFTIQ.MFTIQ import MFTIQ

code_path = Path(__file__).parent.resolve()
repo_path = code_path.parent.resolve()

__version__ = '1.0.0'
__all__ = ['MFTIQ', 'code_path', 'repo_path']
