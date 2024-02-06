import openai
import requests
import cv2

import numpy as np
from io import BytesIO
from PIL import Image, ImageFont, ImageDraw
from matplotlib import font_manager


def genImage(prompt, img_name):
    client = openai.OpenAI()
    try:
        # "My prompt has full detail so no need to add more: ... your prompt here"
        pre_prompt = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"
        final_prompt = pre_prompt + prompt
        response = client.images.generate(
            model="dall-e-3",
            prompt=final_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        img_response = requests.get(response.data[0].url)
        print("revised prompt: ", response.data[0].revised_prompt)
        img = Image.open(BytesIO(img_response.content))
        img.save(img_name)
    except openai.OpenAIError as e:
        print(e.error)
        raise


def openImage(file):
    return Image.open(file)


def detectTextRegion(img_name):
    image = cv2.imread(img_name)

    # image height and width should be multiple of 32
    imgWidth = 320
    imgHeight = 320
    image = cv2.resize(image, (imgWidth, imgHeight))
    (H, W) = image.shape[:2]
    # cv2_imshow(image)

    # load net
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    outputLayers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    output = net.forward(outputLayers)
    scores = output[0]
    geometry = output[1]
    (numRows, numCols) = scores.shape[2:4]

    # get rectangles
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append(np.array([startX, startY, endX, endY]))
            confidences.append(scoresData[x])

    # draw boxes
    indices = cv2.dnn.NMSBoxes(
        rects, confidences, score_threshold=0.8, nms_threshold=0.6)
    boxes = [rects[i] for i in range(len(rects))]
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # cv2.imshow('', image)
    # cv2.waitKey(0)
    cv2.imwrite("detected-"+img_name, image)


def addTextInImage(img_name, text):
    # Call draw Method to add 2D graphics in an image
    img = openImage(img_name)
    drw = ImageDraw.Draw(img)

    # Custom font style and font size
    font = font_manager.FontProperties(family='sans-serif', weight='bold')
    file = font_manager.findfont(font)
    myFont = ImageFont.truetype(file, 65)

    # Add Text to an image
    drw.text((10, 10), text, font=myFont, fill=(255, 0, 0))
    img.save("text-"+img_name)


if __name__ == "__main__":
    img_name = "test1.jpg"
    # genImage("spell the word 'mississipi'")
    # detectTextRegion(img_name)
    addTextInImage(img_name, "mississippi")
