import git
from typing import Tuple, Optional


def get_git_hash() -> Tuple[Optional[str], Optional[str]]:
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.active_branch.object.name, repo.active_branch.commit.hexsha
    except git.exc.InvalidGitRepositoryError:
        return None, None


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
