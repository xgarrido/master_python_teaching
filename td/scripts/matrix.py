import numpy as np
v = np.arange(0, 10)
v[v%2 == 1] = -1

import numpy as np
M = np.ones((4,4))
M[2,3] = 2
M[3,1] = 6
print(M)

import numpy as np
M = np.diag([2, 3, 4, 5, 6], k=-1)
M = M[:, :5]
print(M)

import numpy as np
M = np.tile([[4,3], [2, 1]], (2, 3))
print(M)

import numpy as np
v = np.random.uniform(1, 50, 20)
v = np.where(v < 10, 10, np.where(v > 30, 30, v))
