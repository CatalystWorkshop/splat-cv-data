try:
    from PIL import Image
except ImportError:
    import Image
import PIL.ImageOps
import cv2
from skimage import data, img_as_float, io
import numpy
import time
import random
import os
from queue import Queue
from parse_results_screen import is_results_screen, parse_results_screen, get_winner, results_data_to_json
from parse_map_screen import is_map_screen, parse_map_screen, map_data_to_json
from map_results_associate import associate_players, assoc_results_to_json
import threading
# from web_socket_server import resolve_associated_conflicts
import functools
import asyncio
import websockets
from datetime import datetime
import os
print = functools.partial(print, flush=True)

class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);
        # Start the thread to read frames from the video stream
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.frame = None
        self.status = True
        self.thread.start()
        self.now = time.time()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                self.now = time.time()
            time.sleep(.01)
    def finish(self):
        self.capture.release()
        self.thread.join()


async def send_to_socket(json_dump):
    uri = "ws://192.168.100.22:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json_dump)

def save_image(img, now, np=False):
    if not os.path.exists("./events"):
        os.makedirs("./events")
    file = './events/' + now.strftime("%Y-%m-%d_%H_%M_%S.png")
    if np:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = img
    pil_img.save(file)

def log_event(out, now):
    if not os.path.exists("./events"):
        os.makedirs("./events")
    file = './events/' + now.strftime("%Y-%m-%d_%H_%M_%S.txt")
    with open(file, 'w+') as file:
        file.write(out)

def submit_event(json_dump, now):
    log_event(json_dump, now)
    # asyncio.get_event_loop().run_until_complete(send_to_socket(json_dump))

def get_next_map_or_results_img_from_vid(vidcap):
    success = vidcap.grab()
    count = 0
    point_score_time = None
    time_delay_secs = 2
    frame_freq = int(vidcap.get(cv2.CAP_PROP_FPS)) / 2
    while success:
        time_secs = vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if (count % frame_freq) == 0:
            success,image = vidcap.retrieve()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if is_map_screen(image):
                return 'map',image
            if is_results_screen(image):
                if point_score_time and time_secs - point_score_time >= time_delay_secs:
                    return 'results',Image.fromarray(image)
                elif not point_score_time:
                    point_score_time = time_secs
        success = vidcap.grab()
        count += 1
    return None

def get_next_results_img_from_vid(vidcap):
    success = vidcap.grab()
    count = 0
    point_score_time = None
    time_delay_secs = 2
    frame_freq = int(vidcap.get(cv2.CAP_PROP_FPS)) / 2
    while success:
        time_secs = vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if (count % frame_freq) == 0:
            success,image = vidcap.retrieve()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res_screen = is_results_screen(image)
            if res_screen:
                if point_score_time and time_secs - point_score_time >= time_delay_secs:
                    return Image.fromarray(image)
                elif not point_score_time:
                    point_score_time = time_secs
        success = vidcap.grab()
        count += 1
    return None

def get_parsed_results_from_vid(file):
    vidcap = cv2.VideoCapture(file)
    frame_skip_after = 30 * int(vidcap.get(cv2.CAP_PROP_FPS))
    next_res = get_next_map_or_results_img_from_vid(vidcap)
    match_results = []
    while next_res:
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        imgtype,image = next_res
        if imgtype == 'map':
            map_data = parse_map_screen(image)
            res_img = get_next_results_img_from_vid(vidcap)
            if not res_img:
                vidcap.release()
                return match_results
            res_data = parse_results_screen(res_img)
            assoc_res = associate_players(map_data, res_data)
            print(assoc_results_to_json(assoc_res, res_img))
            # results = resolve_associated_conflicts(player_list, res_img, assoc_res)
        elif imgtype == 'results':
            results = parse_results_screen(image)
        match_results.append(results)
        # Once we find a results screen, skip ahead so it's not detected again later.
        curr_frame = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + frame_skip_after)
        next_res = get_next_map_or_results_img_from_vid(vidcap)
    vidcap.release()
    return match_results

def parse_results_from_stream(src):
    stream = VideoStreamWidget(src)
    data = {}
    mode = 'scan_all'
    wait_start_ts = 0
    time.sleep(1)
    while stream.status:
        img = stream.frame
        now_ts = stream.now
        datetime_now = datetime.now()
        if mode == 'scan_all':
            if is_map_screen(img):
                print('Found map screen')
                print('Parsing map screen...')
                data['map'] = parse_map_screen(img)
                print('Submitting map data to socket')
                save_image(img, datetime_now, np=True)
                submit_event(map_data_to_json(data['map']), datetime_now)
                mode = 'scan_results'
            elif is_results_screen(img):
                print('Found results screen without map screen')
                wait_start_ts = time.time()
                mode = 'wait_results'
        elif mode == 'scan_results':
            if is_results_screen(img):
                print('Found results screen')
                wait_start_ts = time.time()
                mode = 'wait_results'
        elif mode == 'wait_results' and now_ts - wait_start_ts >= 2.5:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            save_image(pil_img, datetime_now, np=False)
            print('Parsing results screen...')
            res_data = parse_results_screen(pil_img)
            if 'map' in data:
                map_data = data['map']
                print('Associating map and results...')
                del data['map']
                assoc_res = associate_players(map_data, res_data)
                print('Submitting results data to socket')
                submit_event(assoc_results_to_json(assoc_res, pil_img.crop((640, 0, 1280, 720))), datetime_now)
            else:
                log_event(results_data_to_json(res_data), datetime_now)
                print('Logged results without map')
            time.sleep(30)
            mode = 'scan_all'
        time.sleep(0.25)

    stream.finish()

def show_webcam(src, mirror=False):
    cam = cv2.VideoCapture(src)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280);
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720);
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # CHANGE ARGUMENT TO MATCH VIRTUAL CAMERA NUMBER
    virtual_cam_id = 1
    parse_results_from_stream(virtual_cam_id)

    # For debugging, show the current webcam instead
    # show_webcam(virtual_cam_id)