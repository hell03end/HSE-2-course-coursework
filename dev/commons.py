import logging


# @TODO: use logging to file
# @TODO: use names of classes or manual names instead of __name__
# @FIXME: correct work of logging level
def enable_logging(name=None, level="info"):
    # @TODO: pass filename, filemode
    _level = logging.INFO
    if level[0] == 'd':
        _level = logging.DEBUG
    elif level[0] == 'w':
        _level = logging.WARNING
    elif level[0] == 'e':
        _level = logging.ERROR
    elif level[0] == 'c':
        _level = logging.CRITICAL
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s"
                               " - %(message)s")
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(__name__)
    logger.setLevel(_level)
    return logger
