class Config(object):
    Device = 'gpu'


class TrainingConfig(object):
    Batch_Size = 128
    Learning_Rate = 2.5e-4
    EveryStepsToCheckLoss = 20
    EveryStepsToSnapshot = 1000
    StepToDecay = 7000
    StepToFinish = 10000


class TestingConfig(object):
    Batch_Size = 128
