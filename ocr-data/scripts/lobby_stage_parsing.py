import cv2
from skimage import io
from skimage.color import rgba2rgb, rgb2gray
from scripts.sprite_parsing import Spritesheet, find_most_similar
from scripts.utils import get_resource_abs_path

def np_binarize(img):
    thresh = 0.9
    return 1.0 * (img > thresh)

with open(get_resource_abs_path('stages.csv'), 'r') as f:
    maps_list = f.read().split('\n')
stage_sprites = Spritesheet(sheet=np_binarize(rgb2gray(rgba2rgb(io.imread(get_resource_abs_path('stages_lobby_1col_355x64.png'))))), \
    num_sprites=23, num_cols=1, sprite_width=355, sprite_height=64, sprite_names=maps_list)


def parse_stage(img):
    x,y = 405,500
    width,height = 355,64

    stage_img = np_binarize(rgb2gray(cv2.cvtColor(img[y:y+height, x:x+width], cv2.COLOR_BGR2RGB)))

    (stage_name, prob) = find_most_similar(stage_img, stage_sprites)
    return stage_name if prob > 0.9 else None