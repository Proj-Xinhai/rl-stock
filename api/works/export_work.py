from .load_work import load_work
import os
import io
import zipfile
from typing import Tuple, Union


def export_work(uuid: str) -> Tuple[bool, str, Union[str, bytes]]:
    work = load_work(uuid)
    if work is None:
        return False, 'uuid', f'`{uuid}` not found'

    if work['status'] != 2:
        return False, 'status', 'work is failed or not yet finished'

    # check if result exists
    if not os.path.exists(f'tasks/works/{uuid}'):
        return False, 'result', 'result not found'

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'a', zipfile.ZIP_DEFLATED, False) as zf:
        zf.write(f'tasks/works/{uuid}.json', arcname=f'{uuid}.json')
        for root, dirs, files in os.walk(f'tasks/works/{uuid}'):
            for file in files:
                zf.write(os.path.join(root, file), arcname=os.path.join(root, file)[len('tasks/works') + 1:])

    return True, 'success', buffer.getvalue()


if __name__ == '__main__':
    raise NotImplementedError('This file is not runnable.')
