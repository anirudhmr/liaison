from .base import Logger as BaseLogger

# Should follow baselogger
from .console import Logger as ConsoleLogger
from .tensorplex import Logger as TensorplexLogger
from .down_sampler import Logger as DownSampleLogger
from .no_op import Logger as NoOpLogger
