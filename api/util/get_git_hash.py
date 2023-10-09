import git


def get_git_hash():
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        return None


if __name__ == '__main__':
    raise RuntimeError('This module is not runnable')
