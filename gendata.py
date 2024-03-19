import pdb
import csv
import os
import re
from PIL import Image
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial

model = ocr_predictor(pretrained=True)


def ocr(filename):
    doc = DocumentFile.from_images(filename)
    # Analyze
    result = model(doc)

    # result.show(doc)
    # print(result)
    # synthetic_pages = result.synthesize()
    #
    # plt.imshow(synthetic_pages[0])
    # plt.axis('off')
    # plt.show()

    export = result.export()
    # Flatten the export
    page_words = [[word for block in page['blocks'] for line in block['lines']
                   for word in line['words']] for page in export['pages']]
    page_dims = [page['dimensions'] for page in export['pages']]

    # Get the coords in [xmin, ymin, xmax, ymax]
    words_abs_coords = [
        [[word['value'], int(round(word['geometry'][0][0] * dims[1])), int(round(word['geometry'][0][1] * dims[0])),
          int(round(word['geometry'][1][0] * dims[1])), int(round(word['geometry'][1][1] * dims[0]))] for word in words if len(word['value']) > 1]
        for words, dims in zip(page_words, page_dims)
    ][0]

    # get max word box
    # max_word_idx = np.argmax([(x1-x0)*(y1-y0)
    #                           for [_, x0, y0, x1, y1] in words_abs_coords])
    # word, xmin, ymin, xmax, ymax = words_abs_coords[max_word_idx]
    for [word, xmin, ymin, xmax, ymax] in words_abs_coords:
        synthesized = result.synthesize()
        doc[0][ymin:ymax+1, xmin:xmax+1,
               :] = synthesized[0][ymin:ymax+1, xmin:xmax+1, :]

    pil_image = Image.fromarray(doc[0])
    img_name = filename.split("/")[-1]
    new_filename = f"m_{img_name}"
    pil_image.save(new_filename)

    # save coordinates
    coor_filename = "coor.csv"
    with open(coor_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([img_name, words_abs_coords])


dir = "/Users/ray/Desktop/AISpelling"
pattern = re.compile(r'^0_\d+\.jpg$')
for filename in os.listdir(dir):
    if pattern.match(filename):
        if (int(filename.split("/")[-1].split(".")[0]) > 100):
            filename = f"{dir}/{filename}"
            print(filename)
            ocr(filename)
