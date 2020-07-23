import os
import math
import json
import sys
import numpy as np
from skimage import data, img_as_float, io
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage.color import rgba2rgb, rgb2gray
from matplotlib import pyplot as plt
from functools import partial
import time
import random
from datetime import datetime
from scripts.utils.sprite_parsing import Spritesheet, find_most_similar
from scripts.utils.path import get_resource_abs_path


ability_image = get_resource_abs_path('abilities_horiz_64px.png')
with open(get_resource_abs_path('abilities_list.csv'), 'r') as file:
    abilities_list = file.read().split('\n')
weapon_image = get_resource_abs_path('weapon_compact_12col_128px.png')
with open(get_resource_abs_path('weapons_list.csv'), 'r') as file:
    weapons_list = [line.split(',')[0] for line in file.read().split('\n')]

abil_spritesheet = Spritesheet(sheet=rgb2gray(rgba2rgb(io.imread(ability_image))), num_sprites=26, num_cols=26, sprite_width=64, sprite_height=64, sprite_names=abilities_list)
weap_spritesheet = Spritesheet(sheet=rgb2gray(rgba2rgb(io.imread(weapon_image))), num_sprites=139, num_cols=12, sprite_width=128, sprite_height=64, sprite_names=weapons_list)


def detect_weapons_from_map_view(i, top_left_x_alpha, top_left_x_bravo, top_left_y_base, y_spacing, side, img):
    top_left_x = (top_left_x_alpha if i < 4 else top_left_x_bravo)
    top_left_y = top_left_y_base + (i % 4) * y_spacing
    crop_img = rgb2gray(img[top_left_y:top_left_y+side, top_left_x:top_left_x+side])

    return find_most_similar(crop_img, weap_spritesheet)


def detect_abilities_from_map_view(i, top_left_x_alpha, top_left_x_bravo, top_left_y_base, y_spacing, x_spacing, side, img):
    headgear = ["Comeback", "Last-Ditch Effort", "Opening Gambit", "Tenacity"]
    clothing = ["Ability Doubler", "Haunt", "Ninja Squid", "Respawn Punisher", "Thermal Ink"]
    shoes = ["Drop Roller", "Object Shredder", "Stealth Jump"]

    top_left_x = (top_left_x_alpha if i < 4 else top_left_x_bravo)
    top_left_y = top_left_y_base + (i % 4) * y_spacing
    crop_img = rgb2gray(img[top_left_y:top_left_y+side, top_left_x:top_left_x+side])
    abil1 = find_most_similar(crop_img, abil_spritesheet, clothing + shoes)

    top_left_x = (top_left_x_alpha if i < 4 else top_left_x_bravo) + x_spacing
    top_left_y = top_left_y_base + (i % 4) * y_spacing
    crop_img = rgb2gray(img[top_left_y:top_left_y+side, top_left_x:top_left_x+side])
    abil2 = find_most_similar(crop_img, abil_spritesheet, headgear + shoes)

    top_left_x = (top_left_x_alpha if i < 4 else top_left_x_bravo) + 2 * x_spacing
    top_left_y = top_left_y_base + (i % 4) * y_spacing
    crop_img = rgb2gray(img[top_left_y:top_left_y+side, top_left_x:top_left_x+side])
    abil3 = find_most_similar(crop_img, abil_spritesheet, headgear + clothing)

    return (abil1, abil2, abil3)

# Returns data in format ([(weap, (abil1, abil2, abil3))], [weap_score, (abil1_score, abil2_score, abil3_score)])
def get_player_data_from_map_view(img):
    top_left_x_alpha = 148
    top_left_x_alpha_abil = 197
    top_left_x_bravo = 1081
    top_left_x_bravo_abil = 1130
    top_left_y_base = 230
    y_spacing = 80
    ability_x_spacing = 32
    weap_side = 39
    abil_side = 23
    top_left_y_abil_base = 240

    abils = []
    weaps = []
    
    weaps = [detect_weapons_from_map_view(top_left_x_alpha=top_left_x_alpha, top_left_x_bravo=top_left_x_bravo, side=weap_side, y_spacing=y_spacing, top_left_y_base=top_left_y_base, img=img, i=i) for i in range(8)]

    abils = [detect_abilities_from_map_view(top_left_x_alpha=top_left_x_alpha_abil, top_left_x_bravo=top_left_x_bravo_abil, side=abil_side, y_spacing=y_spacing, x_spacing=ability_x_spacing, top_left_y_base=top_left_y_abil_base, img=img, i=i) for i in range(8)]

    return list(zip([x[0] for x in weaps], [(x[0][0], x[1][0], x[2][0]) for x in abils])), list(zip([x[1] for x in weaps], [(x[0][1], x[1][1], x[2][1]) for x in abils]))


def is_map_screen(img):
    base_x_alpha = 58
    base_x_bravo = 1006
    base_y_alpha = 230
    base_y_bravo = 231
    y_spacing = 80
    alpha_width = 11
    bravo_width = 6
    thresh = 20
    thresh_2 = 70

    base_x_alpha_2 = 293
    base_y_alpha_2 = 241
    base_x_bravo_2 = 1225
    base_y_bravo_2 = 241
    width_2 = 17

    for player in range(4):
        #Alpha team testing
        base_x = base_x_alpha
        base_y = base_y_alpha + player * y_spacing
        base_x_2 = base_x_alpha_2
        base_y_2 = base_y_alpha_2 + player * y_spacing

        min_col_test_1 = np.amin(img[base_y:base_y+alpha_width,base_x:base_x+alpha_width])
        max_col_test_2 = np.amax(img[base_y_2:base_y_2+width_2,base_x_2:base_x_2+width_2])

        if min_col_test_1 < 255 - thresh or max_col_test_2 > thresh_2:
            return False

        # Bravo team testing
        base_x = base_x_bravo
        base_y = base_y_bravo + player * y_spacing
        base_x_2 = base_x_bravo_2
        base_y_2 = base_y_bravo_2 + player * y_spacing

        min_col_test_1 = np.amin(img[base_y:base_y+bravo_width,base_x:base_x+bravo_width])
        max_col_test_2 = np.amax(img[base_y_2:base_y_2+width_2,base_x_2:base_x_2+width_2])

        if min_col_test_1 < 255 - thresh or max_col_test_2 > thresh_2:
            return False

    return True


# For best results, takes a 1280 x 720 img.
# Takes in a skimage (np array)
def parse_map_screen(img):
    resized_img = cv2.resize(img, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
    results, scores = get_player_data_from_map_view(resized_img)
    weap_thresh = 0.25
    abil_thresh = 0.25
    unsure_weap = [idx + 1 for idx, score in enumerate(scores) if score[0] < weap_thresh]
    unsure_abil = [idx + 1 for idx, score in enumerate(scores) if min(score[1]) < abil_thresh]
    if len(unsure_weap) > 0:
        print("Unsure of Weapons of Players (1 indexed)")
        print(unsure_weap)
    if len(unsure_abil) > 0:
        print("Unsure of Abilities of Players (1 indexed)")
        print(unsure_abil)
    return results

def map_data_to_json(data):
    jsondata = {'eventSource': 'CV', 'timestamp': time.time(), 'eventType': 'map', 'eventData': {'players':[{'weapon': data[i][0],\
    'headgear': data[i][1][0], 'clothing': data[i][1][1], 'shoes': data[i][1][2]} for i in range(8)]}}
    return json.dumps(jsondata, indent=4)