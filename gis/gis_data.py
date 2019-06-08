import numpy as np
import cv2
import pickle

# bin 파일을 만듭니다.
def createDUMP():
    # 신경망 학습용
    # 여기에 40개의 데이터를 입력해 주세요
    # cv2.resize( cv2.imread('./xd2011.png'), (200, 200 )),
    # cv2.resize( cv2.imread('./xd2011.png'), (200, 200 ))
    # 이런식으로 [] 안에 넣주세요!
    rowX = np.array([cv2.resize( cv2.imread('./xd2011.png'), (200, 200 ))], np.float32)
    rowY = np.array([cv2.resize( cv2.imread('./xd2015.png'), (200, 200 ))], np.float32)

    # 신경망 품질 테스트 용
    # 여기에 나머지 10개의 데이터를 입력해 주세요
    # 똑같이 위와 중복 없이 녛주세요!
    rowX2 = np.array([cv2.resize( cv2.imread('./xd2011.png'), (200, 200 ))], np.float32)
    rowY2 = np.array([cv2.resize( cv2.imread('./xd2015.png'), (200, 200 ))], np.float32)

    
    
    X = rowX.transpose([0,3,1,2]) / 255
    Y = rowY.transpose([0,3,1,2]) / 255
    X_test = rowX.transpose([0,3,1,2]) / 255
    Y_test = rowY.transpose([0,3,1,2]) / 255
    dataList = [X,Y,X_test,Y_test]
    with open('gis_data.bin', 'wb') as f:
        pickle.dump(dataList, f) # 현재 로드된 메모리있는 그대로 만든 데이터를 파일로 저장하는 함수입니다.
    print('ok')

# gis_network 가 이 함수로 만들어진 bin파일을 불러옵니다.
def load():
    with open("gis_data.bin", 'rb') as f:
        return pickle.load(f)
