import cv2
import numpy as np
import utlis
import tkinter as tk
from tkinter import filedialog
import os

########################################################################
pathImage = ""
heightImg = 700
widthImg = 700
questions = 5
choices = 5
ans = [1, 1, 1, 1, 1]
########################################################################


def select_image():
    global pathImage
    pathImage = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image",
                                           filetypes=(("JPEG files", ".jpg"), ("PNG files", ".png"), ("All files", ".")))

    if pathImage:
        process_image()
    else:
        print("Failed Loading the Image")


def process_image():
    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    imgFinal = img.copy()
    # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    # CONVERT IMAGE TO GRAY SCALE
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgCanny = cv2.Canny(imgBlur, 10, 70)  # APPLY CANNY

    # FIND ALL COUNTOURS
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(
        imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # FIND ALL CONTOURS
    # DRAW ALL DETECTED CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
    rectCon = utlis.rectContour(contours)  # FILTER FOR RECTANGLE CONTOURS
    # GET CORNER POINTS OF THE BIGGEST RECTANGLE
    biggestPoints = utlis.getCornerPoints(rectCon[0])
    # GET CORNER POINTS OF THE SECOND BIGGEST RECTANGLE
    gradePoints = utlis.getCornerPoints(rectCon[1])

    if biggestPoints.size != 0 and gradePoints.size != 0:

        # BIGGEST RECTANGLE WARPING
        biggestPoints = utlis.reorder(biggestPoints)  # REORDER FOR WARPING
        cv2.drawContours(imgBigContour, biggestPoints, -1,
                         (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        pts1 = np.float32(biggestPoints)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [
            widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(
            pts1, pts2)  # GET TRANSFORMATION MATRIX
        imgWarpColored = cv2.warpPerspective(
            img, matrix, (widthImg, heightImg))  # APPLY WARP PERSPECTIVE

        # SECOND BIGGEST RECTANGLE WARPING
        cv2.drawContours(imgBigContour, gradePoints, -1,
                         (255, 0, 0), 20)  # DRAW THE BIGGEST CONTOUR
        gradePoints = utlis.reorder(gradePoints)  # REORDER FOR WARPING
        ptsG1 = np.float32(gradePoints)  # PREPARE POINTS FOR WARP
        # PREPARE POINTS FOR WARP
        ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        matrixG = cv2.getPerspectiveTransform(
            ptsG1, ptsG2)  # GET TRANSFORMATION MATRIX
        imgGradeDisplay = cv2.warpPerspective(
            img, matrixG, (325, 150))  # APPLY WARP PERSPECTIVE

        # APPLY THRESHOLD
        imgWarpGray = cv2.cvtColor(
            imgWarpColored, cv2.COLOR_BGR2GRAY)  # CONVERT TO GRAYSCALE
        imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[
            1]  # APPLY THRESHOLD AND INVERSE

        boxes = utlis.splitBoxes(imgThresh)  # GET INDIVIDUAL BOXES
        # cv2.imshow("Split Test ", boxes[3])
        countR = 0
        countC = 0
        # TO STORE THE NON ZERO VALUES OF EACH BOX
        myPixelVal = np.zeros((questions, choices))
        for image in boxes:
            # cv2.imshow(str(countR)+str(countC),image)
            totalPixels = cv2.countNonZero(image)
            myPixelVal[countR][countC] = totalPixels
            countC += 1
            if (countC == choices):
                countC = 0
                countR += 1
        print(myPixelVal)

        # FIND THE USER ANSWERS AND PUT THEM IN A LIST
        myIndex = []
        for x in range(0, questions):
            arr = myPixelVal[x]
            # Greter then 5200 is consided as marked.
            cnt = int(0)

            for v in range(0, choices):
                if arr[v] > 5200:
                    cnt = cnt+1
            if cnt == 1:
                myIndexVal = np.where(arr == np.amax(arr))
                print(myIndexVal)
                myIndex.append(myIndexVal[0][0])
            else:
                myIndex.append(-1)

        # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
        grading = []
        for x in range(0, questions):
            if myIndex[x] == -1:
                grading.append(0)
            elif ans[x] == myIndex[x]:
                grading.append(1)
            else:
                grading.append(0)
        score = (sum(grading)/questions)*100

        # DISPLAYING ANSWERS
        utlis.showAnswers(imgWarpColored, myIndex, grading,
                          ans)  # DRAW DETECTED ANSWERS
        utlis.drawGrid(imgWarpColored)  # DRAW GRID
        # NEW BLANK IMAGE WITH WARP IMAGE SIZE
        imgRawDrawings = np.zeros_like(imgWarpColored)  # BLANK IMAGE

        utlis.showAnswers(imgRawDrawings, myIndex,
                          grading, ans)  # DRAW ON NEW IMAGE

        invMatrix = cv2.getPerspectiveTransform(
            pts2, pts1)  # INVERSE TRANSFORMATION MATRIX
        imgInvWarp = cv2.warpPerspective(
            imgRawDrawings, invMatrix, (widthImg, heightImg))  # INV IMAGE WARP

        # SHOW ANSWERS AND GRADE ON FINAL IMAGE
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
        Final = imgFinal.copy()

        cv2.putText(Final, str(score)+"%", (470, 560),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        # IMAGE ARRAY FOR DISPLAY
        imageArray = ([img, imgGray, imgCanny, imgContours],
                      [imgBigContour, imgThresh, imgWarpColored, Final])

        # LABELS FOR DISPLAY
        lables = [["Original", "Gray", "Edges", "Contours"],
                  ["Biggest Contour", "Threshold", "Warpped", "Final"]]

        stackedImage = utlis.stackImages(imageArray, 0.5, lables)
        cv2.imshow("Result", stackedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Final Result", Final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


window = tk.Tk()
window.title("OMR Scanner")
window.geometry("400x200")

label_heading = tk.Label(window, text="OMR Scanner", font=("Arial", 18))
label_heading.pack(pady=10)

btn_select_image = tk.Button(window, text="Select Image", command=select_image)
btn_select_image.pack(pady=20)

window.mainloop()
