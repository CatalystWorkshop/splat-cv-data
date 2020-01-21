try:
    from PIL import Image
except ImportError:
    import Image
import PIL.ImageOps
from skimage import data, img_as_float, io
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage.color import rgba2rgb, rgb2gray
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
import os, os.path
import time
from torch.utils.data import Dataset, DataLoader
import random
import cv2
import functools
from queue import Queue
from multiprocessing import Pool, cpu_count
from collections import namedtuple
print = functools.partial(print, flush=True)


UnassocMatchResults = namedtuple('UnassocMatchResults', 'winner data')
Result = namedtuple('Result', 'weapon ka_count special_count special')
class Spritesheet:
    def __init__(self, sheet, num_sprites, num_cols, sprite_width, sprite_names):
        self.sheet = sheet
        self.num_sprites = num_sprites
        self.num_cols = num_cols
        self.sprite_width = sprite_width
        self.sprite_names = sprite_names
        self.cached_sprites = {}

    def get_sprite(self, idx):
        if idx < 0 or idx >= self.num_sprites:
            print("error, invalid sprite number")
            return ("", [])
        if idx not in self.cached_sprites.keys():
            x = self.sprite_width * (idx % self.num_cols)
            y = self.sprite_width * int(idx / self.num_cols)
            self.cached_sprites[idx] = (self.sprite_names[idx], self.sheet[y:y+self.sprite_width, x:x+self.sprite_width])
        return self.cached_sprites[idx]

# Takes in an RGB image
def find_most_similar(in_img, spritesheet, permit_list=None):
    max_score = 0
    max_img = ''
    max_idx = 0
    for idx in range(spritesheet.num_sprites):
        (sprite_name, sprite) = spritesheet.get_sprite(idx)
        if permit_list and sprite_name not in permit_list:
            continue
        ref_img = resize(sprite, (in_img.shape[0], in_img.shape[1]))

        ssim_val = ssim(ref_img, in_img, data_range=in_img.max() - in_img.min())

        if ssim_val > max_score:
            max_score = ssim_val
            max_img = sprite_name

    return (max_img, round(max_score, 5))

def has_digit(img):
    bbox = img.getbbox()
    if not bbox:
        return False
    return (bbox[2] - bbox[0] > img.size[0] / 6) and (bbox[3] - bbox[1] > img.size[1] / 6)

def predict_digit(img, model):
    trans = transforms.ToTensor()
    img = trans(img)
    img = img.view(1, 1944)
    with torch.no_grad():
        logps = model(img)
    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    maxprob = max(probab)
    predicted = probab.index(max(probab))
    return str(predicted), maxprob

def get_contours_from_image(img):
    img = img.resize((img.size[0] * 3, img.size[1] * 3), PIL.Image.LANCZOS)
    open_cv_img = np.array(img) 
    # Convert RGB to BGR 
    open_cv_img = open_cv_img[:, :, ::-1].copy()
    grey = cv2.cvtColor(open_cv_img, cv2.COLOR_BGR2GRAY)
    grey = cv2.bitwise_not(grey)
    thresh = 50
    (thresh2, grey) = cv2.threshold(grey, thresh, 255, cv2.THRESH_BINARY)
    grey = cv2.bilateralFilter(grey, 11, 17, 17)
    grey = cv2.Canny(grey, 30, 200)
    return Image.fromarray(grey)

def get_data_from_results_view(i, top_left_x_alpha, top_left_x_bravo, top_left_y_alpha, top_left_y_bravo, x_spacing, y_spacing, side, img, spritesheet, permit_lists=None):
    top_left_x = (top_left_x_alpha + (i % 4) * x_spacing) if i < 4 else (top_left_x_bravo - (i % 4) * x_spacing)
    top_left_y = (top_left_y_alpha if i < 4 else top_left_y_bravo) + (i % 4) * y_spacing
    crop_img = rgb2gray(img[top_left_y:top_left_y+side, top_left_x:top_left_x+side])

    permit_list = None
    if permit_lists:
        permit_list = permit_lists[i]
    return find_most_similar(crop_img, spritesheet, permit_list=permit_list)

def detect_weapons_from_results_view(img, specs):
    weapon_image = '../sprites-models/weapon_compact_12col_128px.png'
    with open('../sprites-models/weapons_list.csv', 'r') as file:
        lines = file.read().split('\n')
        weapons_list = [x.split(',')[0] for x in lines]
        weapon_spec_list = [x.split(',')[1] for x in lines]
    weap_spritesheet = Spritesheet(sheet=rgb2gray(rgba2rgb(io.imread(weapon_image))), num_sprites=139, num_cols=12, sprite_width=128, sprite_names=weapons_list)
    top_left_x_alpha = 831
    top_left_x_bravo = 835
    top_left_y_alpha = 128
    top_left_y_bravo = 452

    weap_side = 45
    weap_spacing_y = 55
    weap_spacing_x = 2
    permit_lists = [[weap for i, weap in enumerate(weapons_list) if weapon_spec_list[i] == specs[player]] for player in range(8)]
    weaps = [get_data_from_results_view(i=i, top_left_x_alpha=top_left_x_alpha, top_left_x_bravo=top_left_x_bravo, \
            side=weap_side, y_spacing=weap_spacing_y, x_spacing=weap_spacing_x, top_left_y_alpha=top_left_y_alpha, top_left_y_bravo=top_left_y_bravo, \
            img=img, spritesheet=weap_spritesheet, permit_lists=permit_lists) for i in range(8)]
    # Did you know the * operator is called the Splat operator? Pretty neato
    return zip(*weaps)

def detect_specials_from_results_view(img):
    spec_image = '../sprites-models/specials_horiz_64px_15.png'
    with open('../sprites-models/specials_list.csv', 'r') as f:
        spec_list = f.read().split('\n')
    spec_spritesheet = Spritesheet(sheet=rgb2gray(rgba2rgb(io.imread(spec_image))), num_sprites=15, num_cols=15, sprite_width=64, sprite_names=spec_list)
    top_left_x_alpha = 1183
    top_left_x_bravo = 1189
    top_left_y_alpha = 115
    top_left_y_bravo = 466

    spec_side = 25
    spec_spacing_y = 55
    spec_spacing_x = 2

    specs = [get_data_from_results_view(i=i, top_left_x_alpha=top_left_x_alpha, top_left_x_bravo=top_left_x_bravo,\
         side=spec_side, y_spacing=spec_spacing_y, x_spacing=spec_spacing_x, top_left_y_alpha=top_left_y_alpha, top_left_y_bravo=top_left_y_bravo, \
         img=img, spritesheet=spec_spritesheet) for i in range(8)]
    # If the one in the above function is a splat, then I guess that makes this one Splat 2.
    return zip(*specs)

# 1280 x 720 img
def detect_stats_from_results_view(img):
    input_size = 1944
    hidden_sizes = [128, 64]
    output_size = 10
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size),
                          nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load('../sprites-models/splatfont_digits_model.pt'))
    model.eval()

    results = [(0,0)] * 8
    scores = [(0,0)] * 8
    y_init_t1 = 139
    y_init_t2 = 488
    x_init_t1 = 1148
    x_init_t2 = 1152
    x_spec_offset = 42
    x_init_special_t1 = 1192

    width = 24
    height = 18
    x_spacing = 2
    y_spacing = 55
    thresh = 85

    for i in range(4):
        # Player i
        tl_x = x_init_t1 + x_spacing * i
        tl_y = y_init_t1 + y_spacing * i

        k_a_img = img.crop((tl_x, tl_y, tl_x + width, tl_y + height))
        spec_img = img.crop((tl_x + x_spec_offset, tl_y - int((4 - i) / 2), tl_x + width + x_spec_offset, tl_y + height - int((4 - i) / 2)))

        k_a_grey = get_contours_from_image(k_a_img)
        k_a_grey_1 = k_a_grey.crop((0, 0, int(k_a_grey.size[0] / 2), k_a_grey.size[1]))
        k_a_grey_2= k_a_grey.crop((int(k_a_grey.size[0] / 2), 0, k_a_grey.size[0], k_a_grey.size[1]))
        k_a = ''
        k_a_maxprob_1 = 1
        if has_digit(k_a_grey_1):
            k_a, k_a_maxprob_1 = predict_digit(k_a_grey_1, model)
        k_a_2, k_a_maxprob_2 = predict_digit(k_a_grey_2, model)
        k_a = k_a + k_a_2


        spec_grey = get_contours_from_image(spec_img)
        spec_grey_1 = spec_grey.crop((0, 0, int(spec_grey.size[0] / 2), spec_grey.size[1]))
        spec_grey_2 = spec_grey.crop((int(spec_grey.size[0] / 2), 0, spec_grey.size[0], spec_grey.size[1]))
        spec = ''
        spec_maxprob_1 = 1
        if has_digit(spec_grey_1):
            spec, spec_maxprob_1 = predict_digit(spec_grey_1, model)
        spec_2, spec_maxprob_2 = predict_digit(spec_grey_2, model)
        spec = spec + spec_2
        results[i] = (int(k_a), int(spec))
        scores[i] = (min(k_a_maxprob_1, k_a_maxprob_2), min(spec_maxprob_1, spec_maxprob_2))

        # Player i + 4
        tl_x_t2 = x_init_t2 - x_spacing * i
        tl_y_t2 = y_init_t2 + y_spacing * i

        k_a_img = img.crop((tl_x_t2, tl_y_t2, tl_x_t2 + width, tl_y_t2 + height))
        spec_img = img.crop((tl_x_t2 + x_spec_offset, tl_y_t2 + i, tl_x_t2 + width + x_spec_offset, tl_y_t2 + height + i))

        k_a_grey = get_contours_from_image(k_a_img)
        k_a_grey_1 = k_a_grey.crop((0, 0, int(k_a_grey.size[0] / 2), k_a_grey.size[1]))
        k_a_grey_2= k_a_grey.crop((int(k_a_grey.size[0] / 2), 0, k_a_grey.size[0], k_a_grey.size[1]))
        k_a = ''
        k_a_maxprob_1 = 1
        if has_digit(k_a_grey_1):
            k_a, k_a_maxprob_1 = predict_digit(k_a_grey_1, model)
        k_a_2, k_a_maxprob_2 = predict_digit(k_a_grey_2, model)
        k_a = k_a + k_a_2

        spec_grey = get_contours_from_image(spec_img)
        spec_grey_1 = spec_grey.crop((0, 0, int(spec_grey.size[0] / 2), spec_grey.size[1]))
        spec_grey_2 = spec_grey.crop((int(spec_grey.size[0] / 2), 0, spec_grey.size[0], spec_grey.size[1]))
        spec = ''
        spec_maxprob_1
        if has_digit(spec_grey_1):
            spec, spec_maxprob_1 = predict_digit(spec_grey_1, model)
        spec_2, spec_maxprob_2 = predict_digit(spec_grey_2, model)
        spec = spec + spec_2

        results[i + 4] = (int(k_a), int(spec))
        scores[i + 4] = (min(k_a_maxprob_1, k_a_maxprob_2), min(spec_maxprob_1, spec_maxprob_2))
    return results, scores


# Returns 'alpha' if the spectated player won, 'bravo' otherwise
# Uses the point score in the top left - winners > 1000p, losers < 1000p
def get_winner(img):
    if has_digit(get_contours_from_image(img.crop((46, 39, 46 + 35 , 39 + 48)))):
        return 'alpha'
    return 'bravo'


# For best results, takes a 1280 x 720 img. Otherwise resizes it.
# Takes in a PIL image
#Returns data in format: ('alpha/bravo',[(weap, ka_cnt, spec_cnt, spec)])
def parse_results_screen(img):
    if img.size[0] != 1280 or img.size[1] != 720:
        print('resizing image')
        img = img.resize((1280, 720))
    stats, stat_scores = detect_stats_from_results_view(img)
    specs, spec_scores = detect_specials_from_results_view(np.array(img))
    weaps, weap_scores = detect_weapons_from_results_view(np.array(img), specs=specs)
    k_a_count_thresh = 0.80
    spec_count_thresh = 0.80
    spec_thresh = 0.22
    weap_thresh = 0.22
    unsure_k_a_count = [idx + 1 for idx, score in enumerate(stat_scores) if score[0] < k_a_count_thresh]
    unsure_spec_count = [idx + 1 for idx, score in enumerate(stat_scores) if score[1] < spec_count_thresh]
    unsure_spec = [idx + 1 for idx, score in enumerate(spec_scores) if score < spec_thresh]
    unsure_weap = [idx + 1 for idx, score in enumerate(weap_scores) if score < weap_thresh]
    
    if len(unsure_k_a_count) > 0:
        print("Unsure of K+A of Players (1 indexed)")
        print(unsure_k_a_count)
        print([scores[0] for idx,scores in enumerate(stat_scores) if idx + 1 in unsure_k_a_count])
    if len(unsure_spec_count) > 0:
        print("Unsure of Special Count of Players (1 indexed)")
        print(unsure_spec_count)
        print([scores[1] for idx,scores in enumerate(stat_scores) if idx + 1 in unsure_spec_count])
    if len(unsure_spec) > 0:
        print("Unsure of Specials + Weapons of Players (1 indexed)")
        print(unsure_spec)
    if len(unsure_weap) > 0:
        print("Unsure of Weapons of Players (1 indexed)")
        print(unsure_weap)

    return UnassocMatchResults(get_winner(img), [Result(x[2], x[0][0], x[0][1], x[1]) for x in zip(stats, specs, weaps)])

def is_results_screen(img):
    base_x = 35
    base_y = 5
    width = 170
    height = 25

    min_ok_color = 30
    max_ok_color = 40

    arr = img[base_y:base_y+height,base_x:base_x+width]
    min_col = np.amin(arr)
    max_col = np.amax(arr)

    if max_col > max_ok_color or min_col < min_ok_color:
        return False

    base_x = 38
    base_y = 90
    width = 106
    height = 42

    arr = img[base_y:base_y+height,base_x:base_x+width]
    min_col = np.amin(arr)
    max_col = np.amax(arr)

    if max_col > max_ok_color or min_col < min_ok_color:
        return False   
    return True

start_time = time.time()

def main():
    tmpdir = 'C://Users/bijmb/Documents/splatoon related/ocr/testresults.png'
    # print(is_results_screen(io.imread(tmpdir)))
    start_time = time.time()
    for res in parse_results_screen(Image.open(tmpdir)):
        # a, b = res
        # print(a)
        for line in res:
            print(line)
    #results = parse_results_screen(Image.open(tmpdir))
    #print(get_winner(Image.open(tmpdir)))
    #print(is_results_screen(Image.open(tmpdir)))
    print("--- %s seconds ---" % (time.time() - start_time))
    #for res in results:
    #    print(res)


if __name__ == '__main__':
    main()