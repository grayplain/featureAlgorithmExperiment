import cv2 as cv
import random
import csv

from cv2 import xfeatures2d as Feature


# tirasiSIFT("cut_tirasi_1.png")
def tirasiSIFT(imageName,directory="Images"):
    surf = Feature.SURF_create(400)

    image = cv.imread(directory + "/" + imageName)
    kps = surf.detectAndCompute(image, None)

    circleColor =  (0, 0, 255)
    circleRadius = 2
    circleBorderRadius = 1

    i = 0
    for kp in kps[0]:
        ### キーポイントの値は今の所手動でいじる。
        if i > 200:
            break
        cv.circle(image, (int(kp.pt[0]), int(kp.pt[1])), circleRadius, circleColor, circleBorderRadius)
        # cv.circle(image, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size), (255, 0, 0))
        i += 1

    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def saveKeyPointsToFile(fileName,keyPoints,directory="Images"):
    file = open(directory + "/" + fileName + "-keyPoints.csv",'a')
    csvWriter = csv.writer(file)

    listData = []
    listData.append("angle")
    listData.append("class_id")
    listData.append("octave")
    listData.append("x")
    listData.append("y")
    listData.append("response")
    listData.append("size")
    csvWriter.writerow(listData)

    for keyPoint in keyPoints:
        listData = []
        listData.append(keyPoint.angle)
        listData.append(keyPoint.class_id)
        listData.append(keyPoint.octave)
        listData.append(keyPoint.pt[0])
        listData.append(keyPoint.pt[1])
        listData.append(keyPoint.response)
        listData.append(keyPoint.size)

        csvWriter.writerow(listData)
    file.close()


#tirasiAKAZE("n3big.png",directory="Images/hachi_hachi")
def tirasiAKAZE(imageName,directory="Images", isWriteFile=False):

    #画像準備
    image = cv.imread(directory + "/" + imageName)
    image = cv.resize(image,(500,500))

    #特徴量抽出処理
    akaze = cv.AKAZE_create()
    kp = akaze.detect(image, None)
    kp, des = akaze.compute(image, kp)

    #抽出した特徴量を画像に当て込む
    imgAkaze = cv.drawKeypoints(image,kp,None)

    if isWriteFile:
        randomNumber = random.randint(1,10000000)
        cv.imwrite(directory + '/' + imageName + "-" + str(randomNumber) + ".png", imgAkaze)
        saveKeyPointsToFile(imageName + "-" + str(randomNumber),keyPoints=kp,directory=directory)
    else:
        cv.imshow('image', imgAkaze)
        cv.waitKey(0)
        cv.destroyAllWindows()


def compareSURF(imgPrefName, directory="Images", imgFormatName=".png", allImageNumber=range(1,3)):
    surf = Feature.SURF_create()

    circleColor =  (0, 0, 255)
    circleRadius = 1
    circleBorderRadius = 1

    # 全画像をなめる

    for index in allImageNumber:
        image = cv.imread(directory + "/" + imgPrefName + str(index) + imgFormatName)
        kps = surf.detectAndCompute(image, None)

        i = 0
        for kp in kps[0]:
            if i > 50:
                break
            cv.circle(image, (int(kp.pt[0]), int(kp.pt[1])), circleRadius, circleColor, circleBorderRadius)
            i += 1

        cv.imwrite('result_surf_images/result_image' + str(index) + ".png", image)

#compareSIFT(imgPrefName="n", allImageNumber=range(1, 30), directory="Images/mass_number")
def compareSIFT(imgPrefName, directory="Images", imgFormatName=".png", allImageNumber=range(1,3)):
    sift = Feature.SIFT_create()

    circleColor =  (0, 0, 255)
    circleRadius = 1
    circleBorderRadius = 1

    # 全画像をなめる

    for index in allImageNumber:
        image = cv.imread(directory + "/" + imgPrefName + str(index) + imgFormatName)
        kps = sift.detectAndCompute(image, None)

        i = 0
        for kp in kps[0]:
            if i > 50:
                break
            cv.circle(image, (int(kp.pt[0]), int(kp.pt[1])), circleRadius, circleColor, circleBorderRadius)
            i += 1

        cv.imwrite('result_sift_images/result_image' + str(index) + ".png", image)

#compareAKAZE(imgPrefName="n", allImageNumber=range(1, 30), directory="Images/mass_number")
def compareAKAZE(imgPrefName, directory="Images", imgFormatName=".png", allImageNumber=range(1,3)):
    akaze = cv.AKAZE_create()

    circleColor =  (0, 0, 255)
    circleRadius = 1
    circleBorderRadius = 1

    # 全画像をなめる

    for index in allImageNumber:
        image = cv.imread(directory + "/" + imgPrefName + str(index) + imgFormatName)
        kps = akaze.detectAndCompute(image, None)

        i = 0
        for kp in kps[0]:
            if i > 50:
                break
            cv.circle(image, (int(kp.pt[0]), int(kp.pt[1])), circleRadius, circleColor, circleBorderRadius)
            i += 1

        cv.imwrite('result_akaze_images/result_image' + str(index) + ".png", image)


if __name__ == "__main__":
    images =  ["n1","n2","n3"]
    for image in images:
        tirasiAKAZE(image + ".png",directory="Images/hachi_hachi",isWriteFile=True)
    pass
