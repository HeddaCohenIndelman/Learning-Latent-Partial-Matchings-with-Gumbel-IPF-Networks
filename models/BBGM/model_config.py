from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# BBGM model options
__C.BBGM = edict()
__C.BBGM.SOLVER_NAME = 'LPMP_MGM'
__C.BBGM.LAMBDA_VAL = 80.0
__C.BBGM.SOLVER_PARAMS = edict()
__C.BBGM.SOLVER_PARAMS.timeout = 1000
__C.BBGM.SOLVER_PARAMS.primalComputationInterval = 10
__C.BBGM.SOLVER_PARAMS.maxIter = 100
__C.BBGM.FEATURE_CHANNEL = 1024

#for MGM
__C.BBGM.SOLVER_PARAMS.primalCheckingTriplets = 100
__C.BBGM.SOLVER_PARAMS.presolveIterations = 30
__C.BBGM.SOLVER_PARAMS.multigraphMatchingRoundingMethod = "MCF_PS"
__C.BBGM.SOLVER_PARAMS.tighten = ''
__C.BBGM.SOLVER_PARAMS.tightenIteration = 50
__C.BBGM.SOLVER_PARAMS.tightenInterval = 20
__C.BBGM.SOLVER_PARAMS.tightenConstraintsPercentage = 0.1
__C.BBGM.SOLVER_PARAMS.tightenReparametrization = 'uniform:0.5'

