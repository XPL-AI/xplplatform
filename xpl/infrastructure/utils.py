import sys
import time
import logging
from functools import wraps

#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.Logger(__file__)

def timed(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    start = time.time()
    result = f(*args, **kwds)
    elapsed = time.time() - start
    # print(f'{f.__name__} took {elapsed} time to finish') 
    # logger.debug(f'{f.__name__} took {elapsed} time to finish') 
    return result
  return wrapper