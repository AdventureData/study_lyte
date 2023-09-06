import logging

def setup_log(debug=False):
    """
    Set up the root logger in python.
    Args:
        debug: Whether to log debug statements
    """
    default = "%(name)s [%(levelname)s] %(message)s"

    level = logging.DEBUG if debug else logging.INFO
    handlers = None
    logging.basicConfig(
        format=default,
        level=level, handlers=handlers)
    # Set all ignored modules to be quiet.
    ignore_modules = ['matplotlib']
    for name in logging.Logger.manager.loggerDict.keys():
        if any([m in name for m in ignore_modules]):
            logger = logging.getLogger(name)
            logger.setLevel(logging.CRITICAL)
