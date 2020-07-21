import cv2
from skimage import data, img_as_float, io
from skimage.color import rgba2rgb, rgb2gray
from scripts.sprite_parsing import Spritesheet, find_most_similar
from scripts.utils import get_resource_abs_path

def np_binarize(img):
    thresh = 0.9
    return 1.0 * (img > thresh)

digit_sprites = Spritesheet(sheet=np_binarize(rgb2gray(rgba2rgb(io.imread(get_resource_abs_path('timer_digits.png'))))), num_sprites=10, num_cols=10, sprite_width=24, sprite_height=38, sprite_names=[i for i in range(10)])

def parse_timer_digits(img):
    x1,x2,x3 = 915,956,980
    y = 58
    width,height = 24,38

    digit1_img = np_binarize(rgb2gray(cv2.cvtColor(img[y:y+height, x1:x1+width], cv2.COLOR_BGR2RGB)))
    digit2_img = np_binarize(rgb2gray(cv2.cvtColor(img[y:y+height, x2:x2+width], cv2.COLOR_BGR2RGB)))
    digit3_img = np_binarize(rgb2gray(cv2.cvtColor(img[y:y+height, x3:x3+width], cv2.COLOR_BGR2RGB)))

    (digit1, d1sim) = find_most_similar(digit1_img, digit_sprites)
    (digit2, d2sim) = find_most_similar(digit2_img, digit_sprites)
    (digit3, d3sim) = find_most_similar(digit3_img, digit_sprites)
    joint_sim = d1sim * d2sim * d3sim
    return f"{digit1}:{digit2}{digit3}" if joint_sim >= 0.55  else None


# def snip_digits(src):
#     total_cnt = 0
#     cam = cv2.VideoCapture(src)
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920);
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080);
#     y = 58
#     x1 = 915
#     x2 = 956
#     x3 = 980
#     width = 24
#     height = 38
#     while True:
#         ret_val, img = cam.read()
#         # cv2.imshow('my webcam', img)
#         if cv2.waitKey(1) == 27: 
#             break  # esc to quit
#         if total_cnt % 60 == 0:
#             img_cnt = int(total_cnt / 60)
#             digit1 = Image.fromarray(cv2.cvtColor(img[y:y+height, x1:x1+width], cv2.COLOR_BGR2RGB))
#             digit2 = Image.fromarray(cv2.cvtColor(img[y:y+height, x2:x2+width], cv2.COLOR_BGR2RGB))
#             digit3 = Image.fromarray(cv2.cvtColor(img[y:y+height, x3:x3+width], cv2.COLOR_BGR2RGB))
#             digit1.save(f"./timer/{img_cnt}_d1.png")
#             digit2.save(f"./timer/{img_cnt}_d2.png")
#             digit3.save(f"./timer_test/{img_cnt}_d3.png")

#         total_cnt += 1
#     cv2.destroyAllWindows()