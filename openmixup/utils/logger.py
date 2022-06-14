import json
import logging
from collections import defaultdict

from mmcv.runner import get_dist_info


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "openmixup".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(__name__.split('.')[0])  # i.e., openmixup
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used. Some
            special loggers are:
            - "root": the root logger obtained with `get_root_logger()`.
            - "silent": no message will be printed.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif logger == 'root':
        _logger = get_root_logger()
        _logger.log(level, msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger != 'silent':
        raise TypeError(
            'logger should be either a logging.Logger object, "root", '
            '"silent" or None, but got {}'.format(logger))


def load_json_log(json_log):
    """load and convert json_logs to log_dicts.

    Args:
        json_log (str): The path of the json log file.

    Returns:
        dict[int, dict[str, list]]:
            Key is the epoch, value is a sub dict. The keys in each sub dict
            are different metrics, e.g. memory, bbox_mAP, and the value is a
            list of corresponding values in all iterations in this epoch.

            .. code-block:: python

                # An example output
                {
                    1: {'iter': [100, 200, 300], 'loss': [6.94, 6.73, 6.53]},
                    2: {'iter': [100, 200, 300], 'loss': [6.33, 6.20, 6.07]},
                    ...
                }
    """
    log_dict = dict()
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # skip lines without `epoch` field
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict
