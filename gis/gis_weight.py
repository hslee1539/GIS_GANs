import numpy as np
import pickle
import os

dirname = os.path.dirname(__file__) + os.path.sep

def initWeight(name):
    conv1_stride = 10
    conv1_f = np.float32(np.random.randn(5, 3, 50, 50)) * np.sqrt(2/(3*200*200))
    conv1_b = np.zeros([5], np.float32)
    fc2_w = np.float32(np.random.randn(5 * 16 * 16, 15, 15))* np.sqrt(2/(5*16*16))
    fc2_b = np.zeros([15 * 15], np.float32)
    fc3_w = np.float32(np.random.randn(15 * 15, 5, 16,16))* np.sqrt(2/(15*15))
    fc3_b = np.zeros([5 * 16 * 16], np.float32)
    fc4_f = np.float32(np.random.randn(5 * 16 * 16, 3, 200, 200)) * np.sqrt(1/(5*16*16))
    fc4_b = np.zeros([3* 200 * 200], np.float32)
    weightList = [name, conv1_stride, conv1_f, conv1_b, fc2_w, fc2_b, fc3_w, fc3_b, fc4_f, fc4_b]
    with open(dirname+"gis_" + name + "_weight.bin", 'wb') as f:
        pickle.dump(weightList, f)
    print("done")

def syncWeight(weightList):
    with open(dirname+"gis_" + weightList[0] + "_weight.bin", 'wb') as f:
        pickle.dump(weightList, f)
    print("done")

def loadWeight(name):
    with open(dirname+"gis_" + name + "_weight.bin", 'rb') as f:
        return pickle.load(f)

