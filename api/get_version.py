import git
from typing import Tuple, Optional


def get_version() -> Tuple[Optional[str], Optional[str]]:
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.active_branch.commit.hexsha, repo.active_branch.name
    except git.exc.InvalidGitRepositoryError:
        return None, None


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
