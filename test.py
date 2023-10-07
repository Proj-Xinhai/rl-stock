import pandas
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from api.util.get_algorithm import get_algorithm
import importlib
from tensorflow.python.summary.summary_iterator import summary_iterator
import datetime
from api.works import get_scaler

if __name__ == '__main__':

    print(get_scaler('01154b6cd8294fc1974bc4a361c160e3'))
    """
    # read tensorboard data from tasks/works/6eddc162c92344dfb66b283a4b0e151b/6eddc162c92344dfb66b283a4b0e151b
    start = datetime.datetime.now()
    event_acc = EventAccumulator('tasks/works/01154b6cd8294fc1974bc4a361c160e3/01154b6cd8294fc1974bc4a361c160e3_1')
    event_acc.Reload()
    # # Show all tags in the log file
    print(event_acc.Tags())
    for tag in event_acc.Tags()['scalars']:
        print(tag)
        print(event_acc.Scalars(tag))
    end = datetime.datetime.now()
    print(end-start)
##
    # # # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    # print(event_acc.Scalars('env/hold'))
    """

    """
    start = datetime.datetime.now()
    for summary in summary_iterator('tasks/works/01154b6cd8294fc1974bc4a361c160e3/01154b6cd8294fc1974bc4a361c160e3_1/events.out.tfevents.1696594891.sakkyoi.33124.0'):
        print(summary)
    end = datetime.datetime.now()
    print(end-start)
    """
