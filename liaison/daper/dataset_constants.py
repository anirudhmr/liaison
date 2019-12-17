DATASET_PATH = {
    'tsp-20': '/data/nms/tfp/datasets/tsp/n-20',
    'tsp-50': '/data/nms/tfp/datasets/tsp/n-50',
    'tsp-100': '/data/nms/tfp/datasets/tsp/n-100',
    'milp-facilities-3': '/data/nms/tfp/datasets/milp/facilities/size-3',
    'milp-facilities-10': '/data/nms/tfp/datasets/milp/facilities/size-10',
}

LENGTH_MAP = {
    'tsp-20': dict(train=1000, valid=100, test=100),
    'tsp-50': dict(train=1000, valid=100, test=100),
    'tsp-100': dict(train=1000, valid=100, test=100),
    'milp-facilities-3': dict(train=1000, valid=128, test=128),
    'milp-facilities-10': dict(train=1000, valid=128, test=128),
}

assert sorted(DATASET_PATH.keys()) == sorted(LENGTH_MAP.keys())
