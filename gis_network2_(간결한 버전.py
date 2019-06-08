import cnn_module as cnn
import numpy as np
import gis_data2 as gis_data
import gis_weight

# 데이터 레이어 생성
dataList = gis_data.load()

dataLayer = cnn.createDatasetLayer(dataList[0], dataList[1])

# 옵티마이저 생성
optimizer = cnn.createAda(0.018)

weightList = gis_weight.loadWeight('data2_1')

#encoder 네트워크 생성
with cnn.NetworkBuilder() as builder:
    builder.createNetwork(6)
    builder.addConv3dLayer(weightList[2], weightList[3], weightList[1], 0, 0)
    builder.addBatchnormLayer()
    builder.addReluLayer()
    builder.addFCLayer(weightList[4], weightList[5])
    builder.addBatchnormLayer()
    builder.addReluLayer()
    encoderNet = builder.getNetwork()

#decoder 네트워크 생성
with cnn.NetworkBuilder() as builder:
    builder.createNetwork(5)
    builder.addFCLayer(weightList[6], weightList[7])
    builder.addBatchnormLayer()
    builder.addReluLayer()
    builder.addFCLayer(weightList[8], weightList[9])
    builder.addSigmoidLayer()
    decoderNet = builder.getNetwork()

#gan 네트워크 생성
with cnn.NetworkBuilder() as builder:
    builder.createNetwork(2)
    builder.addNetworkLayer(incoderNet)
    builder.addNetworkLayer(decoderNet)
    ganNet = builder.getNetwork()

#train 네트워크 생성
with cnn.NetworkBuilder() as builder:
    builder.createNetwork(2)
    builder.addNetworkLayer(ganNet)
    builder.addMeansquareLayer()
    trainNet = builder.getNetwork()


trainNet.setLearningData(dataLayer)
trainNet.initForward()
trainNet.initBackward()

if __name__ == '__main__':
    for i in range(1,1000):
        print('epoch',i)
        while(trainNet.forward(0, 1)>0):
            pass
        print('loss',trainNet.out.scalas[0])
        while(trainNet.backward(0,1)>0):
            pass
        while(trainNet.update(optimizer,0,1)>0):
            pass
