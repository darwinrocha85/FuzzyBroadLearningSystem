from enum import Enum

#type reduction of dimensions
class type_reduction(Enum):
  not_reduction = 1
  take_minimus = 2
  PCA = 3
  tsne= 4
  max_positions = 5
  average=6
  take_maximus = 7
