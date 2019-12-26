DATASET_PATH = {
    'tsp-20': '/data/nms/tfp/datasets/tsp/n-20',
    'tsp-50': '/data/nms/tfp/datasets/tsp/n-50',
    'tsp-100': '/data/nms/tfp/datasets/tsp/n-100',
    'milp-facilities-3': '/data/nms/tfp/datasets/milp/facilities/size-3',
    'milp-facilities-10': '/data/nms/tfp/datasets/milp/facilities/size-10',
    'milp-cauction-10': '/data/nms/tfp/datasets/milp/cauction/size-10',
}

LENGTH_MAP = {
    'tsp-20': dict(train=1000, valid=100, test=100),
    'tsp-50': dict(train=1000, valid=100, test=100),
    'tsp-100': dict(train=1000, valid=100, test=100),
    'milp-facilities-3': dict(train=1000, valid=128, test=128),
    'milp-facilities-10': dict(train=10000, valid=1000, test=1000),
    'milp-cauction-10': dict(train=10000, valid=1000, test=1000),
}

NORMALIZATION_CONSTANTS = {
    'milp-facilities-3':
    dict(constraint_rhs_normalizer=3.8935,
         constraint_coeff_normalizer=8.163645833333334,
         obj_coeff_normalizer=319.93917446956965,
         obj_normalizer=1250.3797640539176),
    'milp-facilities-10':
    dict(constraint_rhs_normalizer=1.738010743801653,
         constraint_coeff_normalizer=0.15006158527422991,
         obj_coeff_normalizer=183.39600975244542,
         obj_normalizer=2633.6883431640854),
    'milp-cauction-10':
    dict(constraint_rhs_normalizer=1.0,
         constraint_coeff_normalizer=1.0,
         obj_coeff_normalizer=251.77,
         obj_normalizer=748.822),
}
assert sorted(DATASET_PATH.keys()) == sorted(LENGTH_MAP.keys())
