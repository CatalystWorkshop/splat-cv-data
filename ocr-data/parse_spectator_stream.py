#!/usr/bin/env python
from functools import partial
try:
    from PIL import Image
except ImportError:
    import Image
import time
import numpy as np
import cv2
import sys
from datetime import datetime
from scripts.sprite_parsing import Spritesheet, find_most_similar
from scripts.video_stream_helper import VideoStreamWidget
from scripts.objective_parsing import ObjectiveTimer
from scripts.timer_parsing import parse_timer_digits
from scripts.mode_parsing import parse_mode
from scripts.lobby_stage_parsing import parse_stage
from scripts.parse_results_screen import is_results_screen_1080p, parse_results_screen, get_winner, results_data_to_json
from scripts.event_logging import log_event, save_image
import functools
print = functools.partial(print, flush=True)

def parse_overhead_spectator_stream(src):
    stream = VideoStreamWidget(src, '1080p')
    time.sleep(1)

    old_ts = stream.now
    last_timer_update_ts = time.time()
    last_mode_read_ts = time.time()
    last_lobby_map_read_ts = time.time()
    score_wait_start_ts = None
    results_wait_start_ts = None
    objective_timer = ObjectiveTimer()
    curr_time_left = "5:00"
    curr_game_mode = None
    curr_stage = None
    game_end_time = None
    while stream.status:
        img = stream.frame
        now_ts = stream.now
        
        objective_timer.update(img, now_ts - old_ts)
        if objective_timer.is_game_start():
            print("Game Start!")
        elif objective_timer.is_game_end():
            print("Game End!")
            score_wait_start_ts = now_ts
            game_end_time = datetime.now()
        if objective_timer.is_objective_control_change():
            print("Objective Control Change!")

        if now_ts - last_timer_update_ts >= 0.3:
            new_time_left = parse_timer_digits(img)
            if new_time_left:
                last_timer_update_ts = now_ts
                curr_time_left = new_time_left

        if  not objective_timer.in_game \
            and not curr_game_mode \
                and now_ts - last_mode_read_ts >= 2.5:
            last_mode_read_ts = now_ts
            new_game_mode = parse_mode(img)
            if new_game_mode:
                curr_game_mode = new_game_mode
                print(f"Mode: {curr_game_mode}")

        if not objective_timer.in_game \
            and not curr_stage \
                and now_ts - last_lobby_map_read_ts >= 2:
            last_lobby_map_read_ts = now_ts
            new_stage = parse_stage(img) 
            if new_stage:
                curr_stage = new_stage  
                print(f"Map: {curr_stage}")

        if not objective_timer.in_game \
            and score_wait_start_ts \
            and now_ts - score_wait_start_ts >= 12:
            save_image(img, game_end_time, suffix='_score', np=True)
            score_wait_start_ts = None

        if not results_wait_start_ts \
            and is_results_screen_1080p(img):
            print('Found results screen')
            results_wait_start_ts = now_ts

        elif results_wait_start_ts \
            and now_ts - results_wait_start_ts >= 2.5:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            save_image(pil_img, game_end_time, suffix='_results', np=False)
            print('Parsing results screen...')
            res_data = parse_results_screen(pil_img)
            log_event(results_data_to_json(res_data, curr_game_mode, curr_stage), game_end_time, suffix='_results')
            print('Logged results without map')
            results_wait_start_ts = None
            curr_game_mode = None
            curr_stage = None   
            time.sleep(10) 

        old_ts = now_ts
        time.sleep(0.15)

    stream.finish()

def main():
    print("Overhead Spectator Cam Parser")
    print("------")
    virtual_cam_id = 1

    if len(sys.argv) > 1:
        virtual_cam_id = int(sys.argv[1])

    print(f"Using Virtual Camera {virtual_cam_id}")

    parse_overhead_spectator_stream(virtual_cam_id)


if __name__ == '__main__':
    main()