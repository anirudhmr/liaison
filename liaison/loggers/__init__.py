from .base import Logger as BaseLogger
# Should follow baselogger
from .console import Logger as ConsoleLogger
from .down_sampler import Logger as DownSampleLogger
from .no_op import Logger as NoOpLogger
from .pipe import AvgLogger as AvgPipeLogger
from .tensorplex import Logger as TensorplexLogger
