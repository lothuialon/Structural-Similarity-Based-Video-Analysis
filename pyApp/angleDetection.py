from skimage.metrics import structural_similarity
import cv2
import numpy as np
import math


def process_video(vid):
    # vid start
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    old_frame = None
    while vid.isOpened():
        ret, frame = vid.read()
        if ret == True:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if old_frame is not None:
                # SSIM
                (score, diff) = structural_similarity(old_frame, gray, full=True)
                diff = (diff * 255).astype("uint8")
                diff_box = cv2.merge([diff, diff, diff])

                thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]

                mask = np.zeros(old_frame.shape, dtype='uint8')
                filled_after = gray.copy()
                arr = []
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    arr.append((x, y))
                    arr.append((x + w, y + h))

                box = cv2.minAreaRect(np.asarray(arr))
                pts = cv2.boxPoints(box)  # 4 outer corners

                if pts[3][0] > pts[1][0]:

                    temp1 = pts[3][0]
                    temp2 = pts[1][0]

                else:

                    temp1 = pts[1][0]
                    temp2 = pts[3][0]

                line = cv2.line(diff_box, (int(temp1)+100, int(pts[1][1])), (int(temp2), int(pts[3][1])), (36, 255, 12), 2)
                line2 = cv2.line(diff_box, (0, 719), (1920, 719), (36, 255, 12), 2)

                slope1 = -((pts[3][1])-(pts[1][1]))/((temp2)-(temp1)+100)

                atan = math.atan(slope1)
                pi = 22/7

                degree = atan * (180 / pi)
                degree = 90-degree

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(diff_box, 'Degree: ' + str(degree), (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)
                for c in contours:
                    area = cv2.contourArea(c)
                    if area > 40:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
                cv2.imshow('diff_box', diff_box)
                cv2.imshow('mask', mask)
            old_frame = gray
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    vid.release()
    cv2.destroyAllWindows()


def main():
    vid = cv2.VideoCapture('')
    process_video(vid)

if __name__ == "__main__":
    main()
