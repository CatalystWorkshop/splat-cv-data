import cv2
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from scripts.sprite_parsing import Spritesheet, find_most_similar
from scripts.utils import get_resource_abs_path

def np_binarize(img):
    thresh = 0.9
    return 1.0 * (img > thresh)

with open(get_resource_abs_path('modes.csv'), 'r') as f:
    modes_list = f.read().split('\n')
mode_sprites = Spritesheet(sheet=np_binarize(rgb2gray(rgba2rgb(io.imread(get_resource_abs_path('modes_pregame_4col_225x210.png'))))), \
    num_sprites=4, num_cols=4, sprite_width=225, sprite_height=210, sprite_names=modes_list)


def parse_mode(img):
    x,y = 840,320
    width,height = 225,210

    mode_img = np_binarize(rgb2gray(cv2.cvtColor(img[y:y+height, x:x+width], cv2.COLOR_BGR2RGB)))

    (mode, prob) = find_most_similar(mode_img, mode_sprites)
    return mode if prob > 0.9 else None