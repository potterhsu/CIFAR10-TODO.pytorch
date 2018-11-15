class Config(object):
    Device = 'gpu'


class TrainingConfig(object):
    Batch_Size = 128
    Learning_Rate = 1e-3
    Momentum = 0.9
    Weight_Decay = 0.0005
    EveryStepsToCheckLoss = 20
    EveryStepsToSnapshot = 1000
    StepToFinish = 10000


class TestingConfig(object):
    Batch_Size = 128
