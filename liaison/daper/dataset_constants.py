from pathlib import Path

DATASET_DIR = '/data/nms/tfp'
if not Path(DATASET_DIR).is_dir():
  DATASET_DIR = '/home/gridsan/addanki/data/nms/tfp/'

DATASET_PATH = {
    'tsp-20': f'{DATASET_DIR}/datasets/tsp/n-20',
    'tsp-50': f'{DATASET_DIR}/datasets/tsp/n-50',
    'tsp-100': f'{DATASET_DIR}/datasets/tsp/n-100',
    'milp-facilities-3': f'{DATASET_DIR}/datasets/milp/facilities/size-3',
    'milp-facilities-10': f'{DATASET_DIR}/datasets/milp/facilities/size-10',
    'milp-facilities-100': f'{DATASET_DIR}/datasets/milp/facilities/size-100',
    'milp-cauction-10': f'{DATASET_DIR}/datasets/milp/cauction/size-10',
    'milp-cauction-100': f'{DATASET_DIR}/datasets/milp/cauction/size-100',
    'milp-cauction-100-filtered': f'{DATASET_DIR}/datasets/milp/cauction/size-100-gap-.01',
    'milp-cauction-mixed': f'{DATASET_DIR}/datasets/milp/cauction/size-mixed-small',
    'milp-cauction-200': f'{DATASET_DIR}/datasets/milp/cauction/size-200',
    'milp-cauction-300': f'{DATASET_DIR}/datasets/milp/cauction/size-300',
    'milp-cauction-300-filtered': f'{DATASET_DIR}/datasets/milp/cauction/size-300-filtered',
    'milp-cauction-300-filtered-2': f'{DATASET_DIR}/datasets/milp/cauction/size-300-filtered-2',
    'milp-cauction-25': f'{DATASET_DIR}/datasets/milp/cauction/size-25',
    'milp-cauction-25-filtered': f'{DATASET_DIR}/datasets/milp/cauction/size-25-filtered',
    'milp-cauction-50-filtered': f'{DATASET_DIR}/datasets/milp/cauction/size-50-filtered',
    'milp-setcover-10': f'{DATASET_DIR}/datasets/milp/setcover/size-10',
    'milp-setcover-100': f'{DATASET_DIR}/datasets/milp/setcover/size-100',
    'milp-indset-10': f'{DATASET_DIR}/datasets/milp/indset/size-10',
    'milp-indset-100': f'{DATASET_DIR}/datasets/milp/indset/size-100-cleaned',
    'milp-bcol': f'{DATASET_DIR}/datasets/milp/mip_datasets/BCOL-CLS/BCOL/capacitated_lot_sizing',
    # 'milp-corlat': f'{DATASET_DIR}/datasets/milp/corlat-2',
    'milp-corlat': f'{DATASET_DIR}/datasets/milp/corlat-2-cleaned',
    'milp-regions': f'{DATASET_DIR}/datasets/milp/mip_datasets/Regions200-mps'
}

LENGTH_MAP = {
    'tsp-20': dict(train=1000, valid=100, test=100),
    'tsp-50': dict(train=1000, valid=100, test=100),
    'tsp-100': dict(train=1000, valid=100, test=100),
    'milp-facilities-3': dict(train=1000, valid=128, test=128),
    'milp-facilities-10': dict(train=10000, valid=1000, test=1000),
    'milp-facilities-100': dict(train=1000, valid=100, test=100),
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
    'milp-cauction-300-filtered-2': dict(train=100, valid=50, test=49),
    'milp-setcover-10': dict(train=10000, valid=1000, test=1000),
    'milp-setcover-100': dict(train=1000, valid=100, test=100),
    'milp-indset-10': dict(train=10000, valid=1000, test=1000),
    'milp-indset-100': dict(train=900, valid=90, test=90),
    # 'milp-corlat': dict(train=256, valid=32, test=32),
    # 'milp-corlat': dict(train=1600, valid=160, test=160),
    'milp-corlat': dict(train=1546, valid=155, test=155),
}

DATASET_INFO_PATH = {
    'milp-cauction-25-filtered': f'{DATASET_DIR}/dataset_infos/milp-cauction-25-filtered',
    'milp-cauction-100-filtered': f'{DATASET_DIR}/dataset_infos/milp-cauction-100-filtered',
    'milp-cauction-300-filtered': f'{DATASET_DIR}/dataset_infos/milp-cauction-300-filtered',
    'milp-cauction-300-filtered-2': f'{DATASET_DIR}/dataset_infos/milp-cauction-300-filtered-2',
    'milp-facilities-100': f'{DATASET_DIR}/dataset_infos/milp-facilities-100',
    'milp-corlat': f'{DATASET_DIR}/dataset_infos/milp-corlat2',
    'milp-indset-100': f'{DATASET_DIR}/dataset_infos/milp-indset-100',
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
    'milp-facilities-100':
    dict(
        constraint_rhs_normalizer=0.20705970002940888,
        constraint_coeff_normalizer=0.16596997354171256,
        obj_coeff_normalizer=114.33171071291811,
        obj_normalizer=18216.969891266457,
        optimal_var_vals=0.011125940594059407,
        constraint_degree_normalizer=101.0,
        cont_variable_normalizer=0.01,
        max_nodes=20387,
        max_edges=92659,
    ),
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
    'milp-indset-100':
    dict(
        constraint_rhs_normalizer=1.0,
        constraint_coeff_normalizer=1.0,
        obj_coeff_normalizer=1.0,
        obj_normalizer=225.9688888888889,
        optimal_var_vals=0.4519377777777778,
        constraint_degree_normalizer=4.198888888888889,
        max_nodes=2621,
        max_edges=11607,
    ),
    'milp-cauction-100-filtered':
    dict(
        constraint_rhs_normalizer=1.0,
        constraint_coeff_normalizer=1.0,
        obj_coeff_normalizer=355.88260533379963,
        obj_normalizer=7332.562648075974,
        optimal_var_vals=0.07080000000000002,
        constraint_degree_normalizer=56.618,
        max_nodes=761,
        max_edges=12472,
    ),
    'milp-cauction-300-filtered':
    dict(
        constraint_rhs_normalizer=1.0,
        constraint_coeff_normalizer=1.0,
        obj_coeff_normalizer=344.7605158638172,
        obj_normalizer=22124.905075999453,
        var_normalizer=0.07412066666666665,
        constraint_degree=61.024,
        max_nodes=2021,
        max_edges=16852,
    ),
    'milp-cauction-300-filtered-2':
    dict(
        constraint_rhs_normalizer=1.0,
        constraint_coeff_normalizer=1.0,
        obj_coeff_normalizer=347.0727424227314,
        obj_normalizer=22010.436985601842,
        var_normalizer=0.07254,
        constraint_degree=61.15,
    ),
    'milp-corlat':
    dict(
        constraint_rhs_normalizer=0.9615834498060918,
        constraint_coeff_normalizer=29.802108041421384,
        obj_coeff_normalizer=5.469714241799341,
        obj_normalizer=82.88486416558861,
        optimal_var_vals=2.4384440843245994,
        constraint_degree_normalizer=101.0,
        cont_variable_normalizer=3.0510305147330308,
        max_nodes=1226,
        max_edges=6458,
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
