import cv2
import numpy as np
import random


def L2_distance(vector1, vector2):
    '''
    #vector1과 vector2의 거리 구하기 (L2 distance)
    #distance 는 스칼라
    #np.sqrt(), np.sum() 를 잘 활용하여 구하기
    #L2 distance를 구하는 내장함수로 거리를 구한 경우 감점
    '''
    distance = distance = np.sqrt(np.sum((vector1 - vector2) ** 2))
    return distance

def feature_matching(img1, img2, RANSAC=False, threshold = 300, keypoint_num = None, iter_num = 500, threshold_distance=10):

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print('kp1 개수 : {}'.format(len(kp1)))
    print('kp2 개수 : {}'.format(len(kp2)))
    distance = []
    for idx_1, des_1 in enumerate(des1):
        dist = []
        for idx_2, des_2 in enumerate(des2):
            dist.append(L2_distance(des_1, des_2))

        distance.append(dist)

    distance = np.array(distance)

    min_dist_idx = np.argmin(distance, axis=1)
    min_dist_value = np.min(distance, axis=1)

    points = []
    for idx, point in enumerate(kp1):
        if min_dist_value[idx] >= threshold:
            continue

        x1, y1 = point.pt
        x2, y2 = kp2[min_dist_idx[idx]].pt

        x1 = int(np.round(x1))
        y1 = int(np.round(y1))

        x2 = int(np.round(x2))
        y2 = int(np.round(y2))
        points.append([(x1, y1), (x2, y2)])


    # no RANSAC
    if not RANSAC:

        A = []
        B = []
        for idx, point in enumerate(points):
            '''
            #ToDo
            #A, B 완성
            # A.append(???) 이런식으로 할 수 있음
            # 결과만 잘 나오면 다른방법으로 해도 상관없음
            '''
            A.append(list(point[0]) + [1, 0, 0, 0])
            A.append([0, 0, 0] + list(point[0]) + [1])
            B.append(point[1][0])
            B.append(point[1][1])

        A = np.array(A)
        B = np.array(B)

        '''
        #ToDo
        #X 완성
        #np.linalg.inv(V) : V의 역행렬 구하는것
        #np.dot(V1, V2) : V1과 V2의 행렬곱
        # V1.T : V1의 transpose
        '''

        inverse = np.linalg.inv(np.dot(A.T, A))
        X = np.dot(inverse, A.T)
        X = np.dot(X, B)

        '''
        # ToDo
        # 위에서 구한 X를 이용하여 M 완성
        '''
        M = list(X) + [0, 0, 1]
        M = np.array(M).reshape((3, 3))
        M_ = np.linalg.inv(M)

        '''
        # ToDo
        # backward 방식으로 dst완성
        '''

        #Backward 방식
        # dst의 크기 결정하기
        # 원본 이미지의 모든점에 대해서 행렬 M에 의해 변환을 한 후 이 변환된 점들의 x,y축에 대해서 범위를 구한다
        # 범위를 구한 것에서 각 축에 대해 최솟값과 최댓값을 구한 후 각 축에 대한 최댓값과 최솟값의 차이를 dst의 크기로 정한다.
        h, w = img1.shape[:2]
        height_vec = []
        width_vec = []

        for row in range(h):
            for col in range(w):
                cor_vec = np.dot(M, np.array([[col, row, 1]]).T)
                x = cor_vec[0, 0]
                y = cor_vec[1, 0]
                height_vec.append(y)
                width_vec.append(x)

        h_max = max(height_vec)
        h_min = min(height_vec)
        w_max = max(width_vec)
        w_min = min(width_vec)
        h_ = int(h_max - h_min)
        w_ = int(w_max - w_min)

        dst = np.zeros((h_, w_, 3))
        w_min = int(w_min)
        h_min = int(h_min)


        # BACKWARD
        # backward warping 진행시 이미지가 잘린 부분을 보정을 한다.
        # 보정의 방식은 다음과 같다 .
        # (x,y) -> (x + w_min, y + h_min)로 평행이동시킨후 이 점에 대해서 역행렬과 행렬곱을 수행한다.
        for row_ in range(h_):
            for col_ in range(w_):
                # bilinear
                vec = np.dot(M_, np.array([[col_+ w_min,row_ + h_min, 1]]).T)
                c = vec[0, 0]
                r = vec[1, 0]

                # 원본 이미지크기의 사이즈를 넘어가는 부분에 대해선 값을 가져 올 수 없으므로
                # 이 부분은 건너뛴다.
                if c < 0 or r < 0:
                    continue
                elif c >= w or r >= h:
                    continue
                else:
                    c_left = int(c)  # 버림
                    c_right = min(int(c + 1), w - 1)  # 올림
                    r_top = int(r)  # 버림
                    r_bottom = min(int(r + 1), h - 1)  # 올림

                    s = c - c_left
                    t = r - r_top

                    intensity = (1 - s) * (1 - t) * img1[r_top, c_left] \
                                + s * (1 - t) * img1[r_top, c_right] \
                                + (1 - s) * t * img1[r_bottom, c_left] \
                                + s * t * img1[r_bottom, c_right]
                    dst[row_, col_] = intensity

        dst = dst.astype(np.uint8)
        return dst


    #use RANSAAC
    else:
        points_shuffle = points.copy()

        inliers = []
        M_list = []
        for i in range(iter_num):
            random.shuffle(points_shuffle)
            three_points = points_shuffle[:3]

            A = []
            B = []
            #3개의 point만 가지고 M 구하기
            for idx, point in enumerate(three_points):
                '''
                #ToDo
                #A, B 완성
                # A.append(???) 이런식으로 할 수 있음
                # 결과만 잘 나오면 다른방법으로 해도 상관없음
                '''
                A.append(list(point[0]) + [1, 0, 0, 0])
                A.append([0, 0, 0] + list(point[0]) + [1])
                B.append(point[1][0])
                B.append(point[1][1])

            A = np.array(A)
            B = np.array(B)
            try:
                '''
                #ToDo
                #X 완성
                #np.linalg.inv(V) : V의 역행렬 구하는것
                #np.dot(V1, V2) : V1과 V2의 행렬곱
                # V1.T : V1의 transpose 단, type이 np.array일때만 가능. type이 list일때는 안됨
                '''
                inverse = np.linalg.inv(np.dot(A.T, A))
                X = np.dot(inverse, A.T)
                X = np.dot(X, B)

            except:
                print('can\'t calculate np.linalg.inv((np.dot(A.T, A)) !!!!!')
                continue

            '''
            # ToDo
            # 위에서 구한 X를 이용하여 M 완성
            '''
            M = list(X) + [0, 0, 1]
            M = np.array(M).reshape((3, 3))
            M_list.append(M)

            count_inliers = 0

            for idx, point in enumerate(points):
                '''
                # ToDo
                # 위에서 구한 M으로(3개의 point로 만든 M) 모든 point들에 대하여 예상 point 구하기
                # 구해진 예상 point와 실제 point간의 L2 distance 를 구해서 threshold_distance보다 작은 값이 있는 경우 inlier로 판단
                '''
                ### (M으로 구한 point)
                X_pred = np.dot(M, np.array([[point[0][0], point[0][1], 1]]).T)
                X_pred = X_pred[:2]
                ###(실제 point)
                X_prime = np.array(point[1])
                if L2_distance(X_pred, X_prime) < threshold_distance:
                    count_inliers += 1

            inliers.append(count_inliers)

        inliers = np.array(inliers)
        max_inliers_idx = np.argmax(inliers)

        best_M = np.array(M_list[max_inliers_idx])

        M = best_M
        M_ = np.linalg.inv(M)


        '''
        # ToDo
        # backward 방식으로 dst완성
        '''
        # dst의 크기 결정 과정
        #원본 이미지의 모든점에 대해서 행렬 M에 의해 변환을 한 후 이 변환된 점들의 x,y축에 대해서 범위를 구한다
        # 범위를 구한 것에서 각 축에 대해 최솟값과 최댓값을 구한 후 각 축에 대한 최댓값과 최솟값의 차이를 dst의 크기로 정한다.
        h, w = img1.shape[:2]
        height_vec = []
        width_vec = []

        for row in range(h):
            for col in range(w):
                cor_vec = np.dot(M, np.array([[col, row, 1]]).T)
                x = cor_vec[0, 0]
                y = cor_vec[1, 0]
                height_vec.append(y)
                width_vec.append(x)

        h_max = max(height_vec)
        h_min = min(height_vec)
        w_max = max(width_vec)
        w_min = min(width_vec)
        h_ = int(h_max - h_min)
        w_ = int(w_max - w_min)

        """
        # 보정 작업을 하기위한 값 구하기
        # 보정 작업은 다음과 같다.
        # 단순히 dst의 (x,y)에 역행렬 M_을 곱해서 원본 이미지에서의 대응된 점을 구한다
        # 이때 대응되는 점을 src의 모든 점에 대해서 각 축의 최소값을 빼준다.
        height_vec = []
        width_vec = []
        dst = np.zeros((h_, w_, 3))
        for row in range(dst.shape[0]):
            for col in range(dst.shape[1]):
                cor_vec = np.dot(M_, np.array([[col, row, 1]]).T)
                x = cor_vec[0, 0]
                y = cor_vec[1, 0]
                height_vec.append(y)
                width_vec.append(x)

        s_h_max = max(height_vec)
        s_h_min = min(height_vec)
        s_w_max = max(width_vec)
        s_w_min = min(width_vec)
        s_h_ = int(s_h_max - s_h_min)
        s_w_ = int(s_w_max - s_w_min)
        """
        # BACKWARD
        for row_ in range(h_):
            for col_ in range(w_):
                # bilinear
                vec = np.dot(M_, np.array([[col_+ w_min, row_ + h_min, 1]]).T)
                c = vec[0, 0]
                r = vec[1, 0]

                # 원본 이미지크기의 사이즈를 넘어가는 부분에 대해선 값을 가져 올 수 없으므로
                # 이 부분은 건너뛴다.
                if c < 0 or r < 0:
                    continue
                elif c >= w or r >= h:
                    continue
                else:
                    c_left = int(c)  # 버림
                    c_right = min(int(c + 1), w - 1)  # 올림
                    r_top = int(r)  # 버림
                    r_bottom = min(int(r + 1), h - 1)  # 올림
                    s = c - c_left
                    t = r - r_top
                    intensity = (1 - s) * (1 - t) * img1[r_top, c_left] \
                                + s * (1 - t) * img1[r_top, c_right] \
                                + (1 - s) * t * img1[r_bottom, c_left] \
                                + s * t * img1[r_bottom, c_right]
                    dst[row_, col_] = intensity
        dst = dst.astype(np.uint8)

    return dst


def main():
    img = cv2.imread('../resources/image/building.jpg')
    img_ref = cv2.imread('../resources/image/building_temp.jpg')

    threshold = 300
    iter_num = 500
    #속도가 너무 느리면 100과 같이 숫자로 입력
    keypoint_num = None
    #keypoint_num = 50
    threshold_distance = 10

    dst_no_ransac = feature_matching(img, img_ref, threshold=threshold, keypoint_num=keypoint_num,
                                     iter_num=iter_num, threshold_distance=threshold_distance)
    dst_use_ransac = feature_matching(img, img_ref, RANSAC=True, threshold=threshold, keypoint_num=keypoint_num,
                                      iter_num=iter_num, threshold_distance=threshold_distance)

    cv2.imshow('No RANSAC' + '201804222', dst_no_ransac)
    cv2.imshow('Use RANSAC' + '201804222', dst_use_ransac)

    cv2.imshow('original image' + '201804222', img)
    cv2.imshow('reference image' + '201804222', img_ref)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()



