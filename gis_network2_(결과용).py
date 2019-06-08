import cnn_module as cnn
import numpy as np
import cv2 #화면 출력 용
import gis.gis_data2 as gis_data
import gis.gis_weight as gis_weight
import threading # 화면 출력


# 연구 결과################

# 히(he) 초기와랑 싸비어(xavier ) 초기화 없으면 loss가 엄청 큰 값에서 출발
# 적용하면 좀더 작은 값부터 시작

# 저 두개를 batch norm이 없으면 loss가 떨어 지지 않음.

# batch norm을 적용해도 학습하다가 로컬 최소에 걸림
# 이때 learning rate을 크게 줘서 좌우로 크게 변화를 줘서 탈출 함

# 이미지를 처리 할때는 Conv와 FC중 Conv가 메모리 효율이 좋음
# 예) 이미지를 3 * 200 * 200을 5 * 16 * 16으로 사이즈를 줄일 때,
#   Conv는 5 * 3 * 50 * 50 * sizeof(float) + 5 * sizeof(float)가 필요하지만,
#   FC는 5 * 16 * 16 * 3 * 200 * 200 * sizeof(float) + 5 * 16 * 16 * sizeof(float)가 필요
#   이를 MB로 환산하면 585.94 MB(FC) vs 0.14MB(Conv3d)
#   FC의 경우 가중치가 데이터보다 더 큰 메모리를 차지함.
#   만약 이미지 크기가 400 * 400 이면... 2GB를 사용함...
#   Gan은 encoder 설계는 conv로 하지만, decoder에서 영상이 사이즈가 늘어나야 되는데,
#   Conv3d로 늘릴수 있는 차원은 채널 밖에 없음.(x,y 차원은 늘리는게 불가능)
#   따라서 Conv3d를 역으로 계산하는 Deconv3d로 해야 함. (순전파 방법은 Conv3d의 역잔파와 똑같고, 역전파는 순전파와 같음)
#   이 과제에는 Deconv3d까지 구현하지 못해서 메모리를 비효율적으로 처리하는 FC를 사용



# 학습은 과정 중요!!!!!##################################

# 몇백만에서 5만까지 학습(encoder & decoder)
# 학습은 X->X
# 몇백만에서 몇 만까지 학습할때 로컬 최소 문제가 흔하게 생김
# 이때 Ada의 학습 둔화 운동량을 초기화 함

# X->Y로 바꾸고 다시 학습 (23만으로 loss가 늘어남)
# 순전파는 모든 네트워크를 사용
# 학습할 때는 decoder만 backward 및 update
# encoder가 X->X에 최적화가 됬다는 것이 포인트!

########################################################

# 데이터 레이어 생성
dataList = gis_data.load()

dataLayer = cnn.createDatasetLayer(dataList[0], dataList[1]) # 40, 3, 200, 200
testDataLayer = cnn.createDatasetLayer(dataList[2], dataList[3]) # 10, 3, 200, 200

# 옵티마이저 생성

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
    incoderNet = builder.getNetwork()

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

print('순전파 할, 데이터를 선택하세요')
print('train 데이타 (아무 글자 + 엔터)')
print('test 데이타 (엔터)')
test = False
if(len(input())>0):
    ganNet.setLearningData(dataLayer)
    out = np.zeros(dataList[1].shape)
    ganNet.getRightTerminal().out.release_deep()
    ganNet.getRightTerminal().out = cnn.Tensor.numpy2Tensor(out)
else:
    ganNet.setLearningData(testDataLayer)
    out = np.zeros(dataList[3].shape)
    ganNet.getRightTerminal().out.release_deep()
    ganNet.getRightTerminal().out = cnn.Tensor.numpy2Tensor(out)
    test = True
ganNet.initForward()
#순전파
print('계산중..')
while(ganNet.forward(0,1)>0):
    print('계산중..')

print("표시할 데이터를 선택하세요")
print("train 0 ~ 39")
print('test 0 ~ 9')
num = int(input())

if test:
    cv2.imshow('X', np.uint8(dataList[2][num].transpose(1,2,0) * 255))
    cv2.imshow('T', np.uint8(dataList[3][num].transpose(1,2,0) * 255))
    cv2.imshow('Y', np.uint8( cv2.normalize(out[num].transpose(1,2,0), None, 0, 255, cv2.NORM_MINMAX)))
else:
    cv2.imshow('X', np.uint8(dataList[0][num].transpose(1,2,0) * 255))
    cv2.imshow('T', np.uint8(dataList[1][num].transpose(1,2,0) * 255))
    cv2.imshow('Y', np.uint8( cv2.normalize(out[num].transpose(1,2,0), None, 0, 255, cv2.NORM_MINMAX)))

