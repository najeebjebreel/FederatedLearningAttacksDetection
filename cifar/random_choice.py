import numpy as np

selected_records = np.random.choice(range(50000), 10000, replace=False)

print(selected_records[:10])

