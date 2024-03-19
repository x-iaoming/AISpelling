import openai
import requests
import cv2
import base64
import os
import re
import threading

from prompts import SHORT_SIGNS4
import numpy as np
from io import BytesIO
from PIL import Image, ImageFont, ImageDraw
from matplotlib import font_manager


def genImage(prompt, img_name, client, num=1, model="dall-e-3"):
    try:
        # "My prompt has full detail so no need to add more: ... your prompt here"
        pre_prompt = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"
        final_prompt = pre_prompt + prompt
        response = client.images.generate(
            model=model,
            prompt=final_prompt,
            size="1024x1024",
            quality="standard",
            n=num,
        )
        for i in range(len(response.data)):
            img_response = requests.get(response.data[i].url)
            print("revised prompt: ", response.data[i].revised_prompt)
            img = Image.open(BytesIO(img_response.content))
            img.save(f"{i}_{img_name}")
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
    boxes = [rects[i] for i in indices]
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # cv2.imshow('', image)
    # cv2.waitKey(0)
    cv2.imwrite("detected-"+img_name, image)

    return boxes


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


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def askImage(img_name, api_key, question):
    # Getting the base64 string
    base64_image = encode_image(img_name)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())
    return response.json()['choices'][0]['message']['content']


def generate_mask_with_boxes(box_coordinates, image_size=(320, 320)):
    # Create a new RGBA image with white background
    mask = Image.new('RGBA', image_size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)

    # Draw transparent boxes on the mask
    for a, b, c, d in box_coordinates:
        draw.rectangle([(a, b), (c, d)], fill=(255, 255, 255, 0))

    mask.save('mask.png')


def editImage(img_name, mask_name, prompt, client):
    response = client.images.edit(
        model="dall-e-2",
        image=open(img_name, "rb"),
        mask=open(mask_name, "rb"),
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    img_response = requests.get(response.data[0].url)
    img = Image.open(BytesIO(img_response.content))
    img.save("edited-"+img_name)


if __name__ == "__main__":
    # img_name = "detected-prompt_improved_test_sign.png"

    # os.environ['OPENAI_API_KEY']
    api_key = ""
    client = openai.OpenAI(api_key=api_key)
    # mississipi = "Create an image of the word \'Mississippi\' spelled out correctly and legibly in a horizontal line using uppercase wooden block letters. \
    #     Each letter block should be evenly spaced and aligned with the others on a smooth, light beige surface, ensuring no shadows or reflections obscure the view. \
    #         The background must be a solid, matte dark grey color to contrast sharply with the wooden blocks, highlighting every detail without any additional letters, \
    #             symbols, or distractions in the frame. All the letters should be in focus, from the double \'M\' at the beginning to the final \'I\', composed with a consistent \
    #                 font type and size for uniformity."  # "Generate an image that clearly shows the correctly spelled word \'Mississippi\' using wooden block letters arranged neatly in a single row with no additional letters or distractions. Each letter should be fully visible and legible against a plain, contrasting background to ensure readability."  # "spell the word 'mississipi'"
    # "A teenage girl holding a clear and legible anti-war protest sign that reads \'Make Art, Not War\' in bold, black letters against a white background."
    # sign = "A crowd holding anti-pollution signs"
    # poster = "a detailed inforgraphic/poster explaining the concept of ecotourism, create a black box around each region where there is supposed to be text"

    prompts = SHORT_SIGNS4
    img_name_counter_start = 181  # TODO: remember to update
    tasks = [threading.Thread(target=genImage(prompt, f"{i+img_name_counter_start}.jpg", client, num=1, model="dall-e-3"))
             for i, prompt in enumerate(prompts)]
    print(len(tasks))

    for task in tasks:
        task.start()

    for task in tasks:
        task.join()

    # detectTextRegion(img_name)
    # addTextInImage(img_name, "mississippi")

    """
    image correction through reprompting
    """
    # qn = "Does this image contain any text? Answer YES or No."
    # rsp = askImage(img_name, api_key, qn)
    # print(rsp)
    # if rsp.lower().startswith("yes"):
    #     qn = "Are all text in the image legible and spelled correctly?. Answer YES or No."
    #     rsp = askImage(img_name, api_key, qn)
    #     if rsp.lower().startswith("no"):
    #         qn = f"Given the prompt for this image is {prompt}, what should be the correct \
    #               text in the image in this context? Put the correct text in double quotation mark"
    #         rsp = askImage(img_name, api_key, qn)
    #         rsp_text = re.findall(r'"([^"]*)"', rsp)[-1]
    #         new_prompt = prompt + "make sure the text in image says " + \
    #             rsp_text + " and is legible and spelled correctly."
    #         genImage(new_prompt, "prompt_improved_"+img_name, client)

    """
    Image edit through boxes
    """
    # boxes = detectTextRegion(img_name)
    # generate_mask_with_boxes(boxes)
    # new_prompt = "A crowd holding anti-pollution signs that says 'reduce emissions'"
    # editImage(img_name, "mask.png", new_prompt, client)

    # img_name = "detected-"+img_name
    # qn = "which green box contain text that is spelled wrongly? and what should be the correct word?"
    # rsp = askImage(img_name, api_key, qn)
    # print(rsp)
