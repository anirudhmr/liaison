DATASET_PATH = {
    'tsp-20': '/data/nms/tfp/datasets/tsp/n-20',
    'tsp-50': '/data/nms/tfp/datasets/tsp/n-50',
    'tsp-100': '/data/nms/tfp/datasets/tsp/n-100',
    'milp-facilities-3': '/data/nms/tfp/datasets/milp/facilities/size-3',
    'milp-facilities-10': '/data/nms/tfp/datasets/milp/facilities/size-10',
    'milp-cauction-10': '/data/nms/tfp/datasets/milp/cauction/size-10',
    'milp-cauction-100': '/data/nms/tfp/datasets/milp/cauction/size-100',
    'milp-cauction-100-filtered': '/data/nms/tfp/datasets/milp/cauction/size-100-gap-.01',
    # 'milp-cauction-100-filtered':
    # '/nobackup/users/addanki/datasets/milp/cauction/size-100-gap-.01',
    'milp-cauction-mixed': '/data/nms/tfp/datasets/milp/cauction/size-mixed-small',
    'milp-cauction-200': '/data/nms/tfp/datasets/milp/cauction/size-200',
    'milp-cauction-300': '/data/nms/tfp/datasets/milp/cauction/size-300',
    'milp-cauction-300-filtered': '/data/nms/tfp/datasets/milp/cauction/size-300-filtered',
    'milp-cauction-25': '/data/nms/tfp/datasets/milp/cauction/size-25',
    'milp-cauction-25-filtered': '/data/nms/tfp/datasets/milp/cauction/size-25-filtered',
    'milp-cauction-50-filtered': '/data/nms/tfp/datasets/milp/cauction/size-50-filtered',
    'milp-setcover-10': '/data/nms/tfp/datasets/milp/setcover/size-10',
    'milp-setcover-100': '/data/nms/tfp/datasets/milp/setcover/size-100',
    'milp-indset-10': '/data/nms/tfp/datasets/milp/indset/size-10',
    'milp-indset-100': '/data/nms/tfp/datasets/milp/indset/size-100',
    'milp-bcol': '/data/nms/tfp/datasets/milp/mip_datasets/BCOL-CLS/BCOL/capacitated_lot_sizing',
    'milp-corlat': '/data/nms/tfp/datasets/milp/corlat',
    'milp-regions': '/data/nms/tfp/datasets/milp/mip_datasets/Regions200-mps'
}

LENGTH_MAP = {
    'tsp-20': dict(train=1000, valid=100, test=100),
    'tsp-50': dict(train=1000, valid=100, test=100),
    'tsp-100': dict(train=1000, valid=100, test=100),
    'milp-facilities-3': dict(train=1000, valid=128, test=128),
    'milp-facilities-10': dict(train=10000, valid=1000, test=1000),
    'milp-cauction-10': dict(train=10000, valid=1000, test=1000),
    'milp-cauction-25': dict(train=2000, valid=100, test=100),
    'milp-cauction-25-filtered': dict(train=10000, valid=1000, test=1000),
    'milp-cauction-50-filtered': dict(train=10000, valid=1000, test=1000),
    'milp-cauction-100': dict(train=1000, valid=100, test=100),
    'milp-cauction-mixed': dict(train=2000, valid=100, test=100),
    'milp-cauction-100-filtered': dict(train=1000, valid=100, test=100),
    'milp-cauction-200': dict(train=1000, valid=100, test=100),
    'milp-cauction-300': dict(train=512, valid=16, test=16),
    'milp-cauction-300-filtered': dict(train=1000, valid=200, test=200),
    'milp-setcover-10': dict(train=10000, valid=1000, test=1000),
    'milp-setcover-100': dict(train=1000, valid=100, test=100),
    'milp-indset-10': dict(train=10000, valid=1000, test=1000),
    'milp-indset-100': dict(train=1000, valid=100, test=100),
    'milp-corlat': dict(train=256, valid=32, test=32),
}

DATASET_INFO_PATH = {
    'milp-cauction-25-filtered': '/data/nms/tfp/dataset_infos/milp-cauction-25-filtered',
    'milp-cauction-100-filtered': '/data/nms/tfp/dataset_infos/milp-cauction-100-filtered',
    'milp-cauction-300-filtered': '/data/nms/tfp/dataset_infos/milp-cauction-300-filtered',
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
         obj_normalizer=748.822,
         max_nodes=74,
         max_edges=816),
    'milp-cauction-100':
    dict(constraint_rhs_normalizer=1.0,
         constraint_coeff_normalizer=1.0,
         obj_coeff_normalizer=331.49,
         obj_normalizer=7413.66,
         max_nodes=705,
         max_edges=7654),
    'milp-cauction-200':
    dict(constraint_rhs_normalizer=1.0,
         constraint_coeff_normalizer=1.0,
         obj_coeff_normalizer=334.,
         obj_normalizer=14864.07),
    'milp-cauction-300':
    dict(constraint_rhs_normalizer=1.0,
         constraint_coeff_normalizer=1.0,
         obj_coeff_normalizer=334.,
         obj_normalizer=14864.07),
    'milp-setcover-10':
    dict(
        constraint_rhs_normalizer=1.0,
        constraint_coeff_normalizer=1.0,
        obj_coeff_normalizer=50.5,
        obj_normalizer=75.21,
        max_nodes=1051,
        max_edges=7000,
    ),
    'milp-indset-10':
    dict(
        constraint_rhs_normalizer=1.0,
        constraint_coeff_normalizer=1.0,
        obj_coeff_normalizer=1.5,
        obj_normalizer=21.14,
        max_nodes=231,
        max_edges=824,
    ),
    'milp-cauction-100-filtered':
    dict(
        constraint_rhs_normalizer=1.0,
        constraint_coeff_normalizer=1.0,
        obj_coeff_normalizer=355.88,
        obj_normalizer=7332.56,
        max_nodes=703,
        max_edges=8062,
    ),
    'milp-corlat':
    dict(
        constraint_rhs_normalizer=0.82,
        constraint_coeff_normalizer=30.086,
        obj_coeff_normalizer=5.518,
        obj_normalizer=86.54,
        max_nodes=1127,
        max_edges=5730,
        constraint_degree_normalizer=101,
        cont_variable_normalizer=100.,
    ),
    'milp-cauction-mixed':
    dict(
        constraint_rhs_normalizer=1.0,
        constraint_coeff_normalizer=1.0,
        obj_coeff_normalizer=305.17,
        obj_normalizer=4692.65,
        max_nodes=694,
        max_edges=6944,
    ),
    'milp-cauction-25':
    dict(
        constraint_rhs_normalizer=1.0,
        constraint_coeff_normalizer=1.0,
        obj_coeff_normalizer=301.82,
        obj_normalizer=1872.97,
        max_nodes=179,
        max_edges=2114,
    ),
}
