from .base import Logger as BaseLogger
# Should follow baselogger
from .console import Logger as ConsoleLogger
from .down_sampler import Logger as DownSampleLogger
from .kv_stream import Logger as KVStreamLogger
from .no_op import Logger as NoOpLogger
from .pipe import AvgLogger as AvgPipeLogger
from .tensorplex import Logger as TensorplexLogger
from .file_stream import Logger as FileStreamLogger
