from import_lib import lib
#from tensor.main_module import *
from cnn.struct.updatelist_module import UpdateList
from ctypes import Structure, c_int, POINTER, c_float, CFUNCTYPE, c_void_p

_cnn_optimizer_fpUpdate = CFUNCTYPE(c_void_p, UpdateList, c_int, c_int)

class _Optimizer(Structure):
    _fields_ = [
        ('learning_raate', c_float),
        # 원본은 c_void_p가 아닌, POINTER(_Optimizer)여야 하지만,
        # 파이썬은 c언어처럼 이름만 선언하는 기능이 없기 때문에
        # c_void_p로 받음.
        ('update', _cnn_optimizer_fpUpdate)
    ]

cnn_optimizer_fpUpdate = CFUNCTYPE(POINTER(_Optimizer), UpdateList, c_int, c_int)

def _getLearningRate(self):
    return self.contents.learning_raate

def _setLearningRate(self, value):
    self.contents.learning_raate = value

def _getUpdate(self):
    return self.contents.update

def _setUpdate(self, value):
    self.contents.update = value

def _update(self, updatelist, index, max_index):
    lib.cnn_optimizer_update(self, updatelist, index, max_index)

def _create(learning_rate, update):
    return lib.cnn_create_optimizer(learning_rate, update)

def _release(self):
    lib.cnn_release_optimizer(self)


lib.cnn_optimizer_update.argtypes = (POINTER(_Optimizer), UpdateList)
lib.cnn_create_optimizer.argtypes = (c_int, cnn_optimizer_fpUpdate)
lib.cnn_create_optimizer.restype = POINTER(_Optimizer)
lib.cnn_release_optimizer.argtypes = [POINTER(_Optimizer)]



Optimizer = POINTER(_Optimizer)
Optimizer.__doc__ = " cnn_Optimizer 구조체의 포인터에 프로퍼티와 메소드를 추가한 클래스입니다."
Optimizer.learning_raate = property(_getLearningRate, _setLearningRate)
Optimizer._update = property(_getUpdate, _setUpdate)
Optimizer.create = staticmethod(_create)
Optimizer.release = _release
Optimizer.update = _update


