import numpy as np

list_items = [['1.jpg', np.eye(1)],
              ['11.jpg', np.eye(2)],
              ['111.jpg', np.eye(3)]]

print(list_items)

list_array = np.asanyarray(list_items, dtype=object)
print(list_array)