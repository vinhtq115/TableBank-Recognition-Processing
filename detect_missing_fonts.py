import os
import cv2
from tqdm.contrib.concurrent import process_map

PATH_TO_IMAGE_FOLDER = r'E:\TableBank-Recognition\Recognition\images'


# PATH_TO_ORIGINAL_ANNOTATIONS = r'C:\Users\starc\PycharmProjects\TableBank-Recognition-Processing\original_annotations'


def check_template(binary, template, threshold=0.85, method=cv2.TM_CCOEFF_NORMED):
    if binary.shape[0] < template.shape[0] or binary.shape[1] < template.shape[1]:
        return 0, binary
    res = cv2.matchTemplate(binary, template, method)
    #temp_image = binary.copy()
    template_height, template_width = template.shape[:2]

    count_match = 0
    max_val = 1.0
    prev_min_val, prev_max_val, prev_min_loc, prev_max_loc = None, None, None, None
    while max_val > threshold:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if prev_min_val == min_val and prev_max_val == max_val and prev_min_loc == min_loc and prev_max_loc == max_loc:
            break
        else:
            prev_min_val, prev_max_val, prev_min_loc, prev_max_loc = min_val, max_val, min_loc, max_loc
        if max_val > threshold:
            count_match += 1
            start_row = max_loc[1] - template_height // 2 if max_loc[1] - template_height // 2 >= 0 else 0
            end_row = max_loc[1] + template_height // 2 + 1 if max_loc[1] + template_height // 2 + 1 <= res.shape[0] else res.shape[0]
            start_col = max_loc[0] - template_width // 2 if max_loc[0] - template_width // 2 >= 0 else 0
            end_col = max_loc[0] + template_width // 2 + 1 if max_loc[0] + template_width // 2 + 1 <= res.shape[1] else res.shape[0]
            res[start_row: end_row, start_col: end_col] = 0
            #temp_image = cv2.rectangle(temp_image, (max_loc[0], max_loc[1]),
            #                           (max_loc[0] + template_width + 1, max_loc[1] + template_height + 1), 0)

    return count_match, None


template = cv2.imread('missing_font_template.png', 0)
template_tl = cv2.imread('missing_font_template_tl.png', 0)
template_tr = cv2.imread('missing_font_template_tr.png', 0)
template_bl = cv2.imread('missing_font_template_bl.png', 0)
template_br = cv2.imread('missing_font_template_br.png', 0)


def det(file):
    image_path = os.path.join(PATH_TO_IMAGE_FOLDER, file)
    img = cv2.imread(image_path, 0)  # Read image as grayscale
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    count_match, _ = check_template(binary, template)
    count_match1, _ = check_template(binary, template)
    count_match2, _ = check_template(binary, template)
    count_match3, _ = check_template(binary, template)
    count_match4, _ = check_template(binary, template)
    if count_match + count_match1 + count_match2 + count_match3 + count_match4 >= 10:
        return [file, True]
    else:
        return [file, False]


if __name__ == '__main__':
    res = []
    for root, _, files in os.walk(PATH_TO_IMAGE_FOLDER):
        res = process_map(det, files, max_workers=12, chunksize=15)
        # for file in files:
        #     image_path = os.path.join(PATH_TO_IMAGE_FOLDER, file)
        #     img = cv2.imread(image_path, 0)  # Read image as grayscale
        #     _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        #     count_match, temp_image = check_template(binary, template)
        #     if count_match >= 2:
        #         # cv2.imshow('Result', temp_image)
        #         # q = cv2.waitKey()
        #         # if q == 32:
        #         counter += 1
        #         print(file, count_match)
        # cv2.destroyAllWindows()
    counter = 0
    list_file = open('rec_deet.txt', 'w')
    for r in res:
        if r[1]:
            counter += 1
            list_file.write(r[0] + '\n')
    list_file.close()
    print(counter)
