import cnn_module as cnn
import numpy as np
import cv2 #화면 출력 용
import gis.gis_data2 as gis_data
import gis.gis_weight as gis_weight
import threading # 화면 출력

# 빠르게 학습하고 싶으면 False로 바꾸세요!.
# 신경망 결과를 그려줍니다.
visible = False

# 연구 결과################

# 히 초기와랑 싸비어 초기화 없으면 loss가 몇싶만 부터 출발함...
# 적용하면 몇천때 부터 시작

# batch norm이 없으면 loss가 떨어 지지 않음.

###########################

# 데이터 레이어 생성
dataList = gis_data.load()

dataLayer = cnn.createDatasetLayer(dataList[0], dataList[1]) # N, 3, 200, 200
print("dataList[0]", dataList[0].shape)

# 옵티마이저 생성
optimizer = cnn.createAda(0.018)

print("가중치 이름을 입력하세요. 만들지 않으면 gis_weight.py를 실행하여 데이터를 만드세요")
weightList = gis_weight.loadWeight(input())

#incoder 네트워크 생성
with cnn.NetworkBuilder() as builder:
    builder.createNetwork(6)
    builder.addConv3dLayer(weightList[2], weightList[3], weightList[1], 0, 0)
    builder.addBatchnormLayer()
    builder.addReluLayer()
    builder.addFCLayer(weightList[4], weightList[5])
    builder.addBatchnormLayer()
    builder.addReluLayer()
    incoderNet = builder.getNetwork() # encoder임

#decoder 네트워크 생성
with cnn.NetworkBuilder() as builder:
    builder.createNetwork(5)
    builder.addFCLayer(weightList[6], weightList[7])
    builder.addBatchnormLayer()
    builder.addReluLayer()
    builder.addFCLayer(weightList[8], weightList[9])
    builder.addSigmoidLayer()
    decoderNet = builder.getNetwork()

#train 네트워크 생성
with cnn.NetworkBuilder() as builder:
    builder.createNetwork(2)
    builder.addNetworkLayer(decoderNet)
    builder.addMeansquareLayer()
    trainNet = builder.getNetwork()

#전체 네트워크 생성
with cnn.NetworkBuilder() as builder:
    builder.createNetwork(2)
    builder.addNetworkLayer(incoderNet)
    builder.addNetworkLayer(trainNet)
    allNet = builder.getNetwork()

allNet.setLearningData(dataLayer)
allNet.initForward()
allNet.initBackward()

if __name__ == '__main__':
    for i in range(1,1000):
        print('epoch',i)
        while(allNet.forward(0, 1)>0):
            pass
        
        print('loss',trainNet.out.scalas[0])

        #allNet이 아닌 decoderNet만 하기 위해 allNet 네부에
        # decoderNet 뒤에 mean square layer가 붙은 tranNet 사용
        while(trainNet.backward(0,1)>0):
            pass
        
        while(trainNet.update(optimizer,0,1)>0):
            pass
        if(i % 15 == 0):
            print("auto save")
            gis_weight.syncWeight(weightList)
