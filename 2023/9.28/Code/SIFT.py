import cv2
import numpy as np
from decimal import Decimal

for i in range(1, 20):

    filename1 = "D:\VSworkSpace\Code\python\Fig1\CameraImage_" + str(i) + ".jpg"
    filename2 = "D:\VSworkSpace\Code\python\Fig1\CameraImage_" + str(i + 1) + ".jpg"

    # 读取两幅图像
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    frame1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    _, frame1 = cv2.threshold(frame1, 30, 255, cv2.THRESH_TOZERO)
    _, frame2 = cv2.threshold(frame2, 30, 255, cv2.THRESH_TOZERO)



    # 获取图像的高度和宽度
    height, width = frame1.shape

    # 创建一个与图像大小相同的掩膜
    mask =  np.zeros((height, width), dtype=np.uint8)

    # 在掩膜上绘制一个白色的矩形，表示要提取特征点的区域
    start_point = (200, 200)  # 矩形左上角坐标
    end_point = (700, 550)    # 矩形右下角坐标
    cv2.rectangle(mask, start_point, end_point, 255, thickness=cv2.FILLED)

    # 使用掩码对原始图像进行抠图
    frame1 = cv2.bitwise_and(frame1, mask)
    frame2 = cv2.bitwise_and(frame2, mask)

    #获得图片的轮廓值
    contours1, _ = cv2.findContours(frame1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours2, _ = cv2.findContours(frame2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #在图片中画出图片的轮廓值
    draw_img1 = img1.copy()
    draw_img2 = img2.copy()
    ret1 = cv2.drawContours(draw_img1, contours1, 0, (0, 0, 255), 2)
    ret2 = cv2.drawContours(draw_img2, contours2, 0, (0, 0, 255), 2)
    cv2.imshow('ret1', ret1)
    cv2.imshow('ret2', ret2)

    # 初始化SIFT特征检测器
    sift = cv2.SIFT_create()


    # 创建一个与图像大小相同的掩膜
    feature_mask1 =  np.zeros((height, width), dtype=np.uint8)
    feature_mask2 =  np.zeros((height, width), dtype=np.uint8)

    
    feature_mask1 = cv2.fillPoly(feature_mask1, contours1, 255)
    feature_mask2 = cv2.fillPoly(feature_mask2, contours2, 255)
    
    cv2.imshow("Object1_mask", feature_mask1)
    cv2.imshow("Object2_mask", feature_mask2)

    # 检测特征点和计算描述子
    keypoints1, descriptors1 = sift.detectAndCompute(frame1, mask = feature_mask1)
    keypoints2, descriptors2 = sift.detectAndCompute(frame2, mask = feature_mask2)

    # 将图片1和2进行深拷贝，用于画出特征点
    show_img1 = img1.copy()
    show_img2 = img2.copy()
    # 绘制特征点在原图副本上
    cv2.drawKeypoints(img1, keypoints1, show_img1)
    cv2.drawKeypoints(img2, keypoints2, show_img2)
    cv2.imshow("show_img1", show_img1)
    cv2.imshow("show_img2", show_img2)
    cv2.imwrite("D:\VSworkSpace\Code\python\Fig1_output\Feature_points\Feature_point_" + str(i) + ".jpg", show_img1)

    # 初始化暴力匹配器
    bf = cv2.BFMatcher()

    # 进行特征点匹配
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)  #knn算法 

    # 应用比例测试，保留良好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 提取匹配点的坐标
    points1 = np.uint16([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.uint16([keypoints2[m.trainIdx].pt for m in good_matches])

    # 根据匹配点计算均值求出质心
    center1 = np.uint16(np.mean(points1, axis = 0))
    center2 = np.uint16(np.mean(points2, axis = 0))


    # 显示匹配结果
    result_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)
    cv2.imshow("Matches", result_img)

    cv2.circle(img1, (center1[0], center1[1]), 6, (0,0,255), -1)
    cv2.circle(img2, (center1[0], center1[1]), 6, (0,0,255), -1)
    cv2.imwrite("D:\VSworkSpace\Code\python\Fig1_output\Centroid\\" + "Centroid_" + str(i+1) + ".jpg", img2)

    # 将与临近两张图特征点匹配图像保存
    cv2.imwrite("D:\VSworkSpace\Code\python\Fig1_output\Matches\Matches" + str(i) + ".jpg",result_img)
  
    # 展示相邻两帧图像的质心
    cv2.imshow("img1_centroid",img1)
    cv2.imshow("img2_centroid",img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()