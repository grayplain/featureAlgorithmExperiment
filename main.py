import cv2 as cv

from cv2 import xfeatures2d as Feature

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

def tirasiAKAZE(imageName,directory="Images"):
    image = cv.imread(directory + "/" + imageName)
    akaze = cv.AKAZE_create()
    kp = akaze.detect(image, None)
    kp, des = akaze.compute(image, kp)

    imgAkaze = cv.drawKeypoints(image,kp,None)
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
    #compareSIFT(imgPrefName="download-",allImageNumber=range(1,3))
    #compareSIFT(imgPrefName="n", allImageNumber=range(1, 30), directory="Images/mass_number")

    #compareAKAZE(imgPrefName="n", allImageNumber=range(1, 30), directory="Images/mass_number")
    tirasiAKAZE("n3big.png",directory="Images/hachi_hachi")
    #tirasiSIFT("cut_tirasi_1.png")
    #tirasiSIFT("cut_tirasi_2.png")
    #tirasiSIFT("cut_tirasi_3.png")
    pass
