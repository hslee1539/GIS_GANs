import numpy as np
import cv2
import pickle
import os

dirname = os.path.dirname(__file__) + os.path.sep

# bin 파일을 만듭니다.
def createDUMP():
    # 신경망 학습용
    # 여기에 40개의 데이터를 입력해 주세요
    # cv2.resize( cv2.imread(dirname + './xd2011.png'), (200, 200 )),
    # cv2.resize( cv2.imread(dirname + './xd2011.png'), (200, 200 ))
    # 이런식으로 [] 안에 넣주세요!
    rowX = np.array([cv2.resize( cv2.imread(dirname + './2013_000.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_001.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_002.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_003.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_004.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_005.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_006.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_008.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_009.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_010.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_011.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_012.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_013.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_014.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_015.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_016.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_017.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_018.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_019.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_020.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_021.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_022.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_023.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_024.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_025.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_026.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_027.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_028.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_029.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_030.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_031.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_032.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_033.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_034.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_035.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_036.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_037.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_038.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_039.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2013_040.png'), (200, 200 ))], np.float32)
    rowY = np.array([cv2.resize( cv2.imread(dirname + './2015_000.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_001.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_002.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_003.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_004.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_005.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_006.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_008.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_009.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_010.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_011.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_012.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_013.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_014.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_015.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_016.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_017.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_018.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_019.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_020.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_021.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_022.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_023.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_024.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_025.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_026.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_027.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_028.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_029.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_030.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_031.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_032.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_033.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_034.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_035.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_036.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_037.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_038.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_039.png'), (200, 200 )),
                     cv2.resize( cv2.imread(dirname + './2015_040.png'), (200, 200 ))], np.float32)

    # 신경망 품질 테스트 용
    # 여기에 나머지 10개의 데이터를 입력해 주세요
    # 똑같이 위와 중복 없이 녛주세요!
    rowX2 = np.array([cv2.resize( cv2.imread(dirname + './2013_041.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2013_042.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2013_043.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2013_044.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2013_045.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2013_046.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2013_047.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2013_048.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2013_049.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2013_050.png'), (200, 200 ))], np.float32)
    rowY2 = np.array([cv2.resize( cv2.imread(dirname + './2015_041.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2015_042.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2015_043.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2015_044.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2015_045.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2015_046.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2015_047.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2015_048.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2015_049.png'), (200, 200 )),
                      cv2.resize( cv2.imread(dirname + './2015_050.png'), (200, 200 ))], np.float32)

    
    
    X = rowX.transpose([0,3,1,2]) / 255
    Y = rowY.transpose([0,3,1,2]) / 255
    X_test = rowX2.transpose([0,3,1,2]) / 255
    Y_test = rowY2.transpose([0,3,1,2]) / 255
    dataList = [X,Y,X_test,Y_test]
    with open(dirname + 'gis_data2.bin', 'wb') as f:
        pickle.dump(dataList, f) # 현재 로드된 메모리있는 그대로 만든 데이터를 파일로 저장하는 함수입니다.
    print('ok')

# gis_network 가 이 함수로 만들어진 bin파일을 불러옵니다.
def load():
    with open(dirname+ "gis_data2.bin", 'rb') as f:
        return pickle.load(f)
