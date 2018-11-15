import numpy as np
confusion_matrix = np.zeros((10, 10))





'''
logits = model.eval().forward(images)
_, predictions = logits.max(dim=1)
num_hits += (predictions == labels).sum().item()
'''