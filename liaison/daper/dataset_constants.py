DATASET_PATH = {
    'tsp-20': '/data/nms/tfp/datasets/tsp/n-20',
    'tsp-50': '/data/nms/tfp/datasets/tsp/n-50',
    'tsp-100': '/data/nms/tfp/datasets/tsp/n-100',
}

LENGTH_MAP = {
    'tsp-20': dict(train=1000, valid=100, test=100),
    'tsp-50': dict(train=1000, valid=100, test=100),
    'tsp-100': dict(train=1000, valid=100, test=100),
}

assert sorted(DATASET_PATH.keys()) == sorted(LENGTH_MAP.keys())
