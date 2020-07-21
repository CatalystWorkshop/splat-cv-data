import numpy as np
import functools
import time
print = functools.partial(print, flush=True)

class UISpacingConfig:
    def __init__(self, alpha_base_x, bravo_base_x, alpha_y1, alpha_y2, bravo_y1, bravo_y2, alpha_spacing, bravo_spacing, alpha_width, bravo_width):
        self.alpha_base_x = alpha_base_x
        self.bravo_base_x = bravo_base_x
        self.alpha_y1 = alpha_y1
        self.alpha_y2 = alpha_y2
        self.bravo_y1 = bravo_y1
        self.bravo_y2 = bravo_y2
        self.alpha_spacing = alpha_spacing
        self.bravo_spacing = bravo_spacing
        self.alpha_width = alpha_width
        self.bravo_width = bravo_width
    
    def __str__(self):
        return f"Alpha Base X: {self.alpha_base_x}, Bravo Base X: {self.bravo_base_x}, \
            Y1: {self.alpha_y1}, Y2: {self.alpha_y2}, \
            Alpha Spacing: {self.alpha_spacing}, Bravo Spacing: {self.bravo_spacing}, \
            Alpha Width: {self.alpha_width}, Bravo Width: {self.bravo_width}"

NO_OBJ_STATE = -99
ALPHA_DANGER_OBJ_STATE = -2
BRAVO_ADV_OBJ_STATE = -1
NEUTRAL_OBJ_STATE = 0
ALPHA_ADV_OBJ_STATE = 1
BRAVO_DANGER_OBJ_STATE = 2

alpha_danger_state = UISpacingConfig(alpha_base_x=600, bravo_base_x=1074, alpha_y1=35, alpha_y2=40, bravo_y1=26, bravo_y2=33, alpha_spacing=77, bravo_spacing=98, alpha_width=16, bravo_width=22)
bravo_adv_state = UISpacingConfig(alpha_base_x=582, bravo_base_x=1080, alpha_y1=32, alpha_y2=39, bravo_y1=28, bravo_y2=35, alpha_spacing=82, bravo_spacing=92, alpha_width=18, bravo_width=22)
neutral_state = UISpacingConfig(alpha_base_x=565, bravo_base_x=1080, alpha_y1=30, alpha_y2=38, bravo_y1=30, bravo_y2=38, alpha_spacing=86, bravo_spacing=86, alpha_width=20, bravo_width=20)
alpha_adv_state = UISpacingConfig(alpha_base_x=548, bravo_base_x=1076, alpha_y1=28, alpha_y2=35, bravo_y1=32, bravo_y2=39, alpha_spacing=92, bravo_spacing=82, alpha_width=22, bravo_width=18)
bravo_danger_state = UISpacingConfig(alpha_base_x=530, bravo_base_x=1076, alpha_y1=26, alpha_y2=33, bravo_y1=35, bravo_y2=40, alpha_spacing=98, bravo_spacing=77, alpha_width=22, bravo_width=16)
state_list = [alpha_danger_state, bravo_adv_state, neutral_state, alpha_adv_state, bravo_danger_state]

def matches_state(img, config):
    max_color = 70
    
    for player in range(4):
        #Alpha team testing
        base_x = config.alpha_base_x + config.alpha_spacing * player

        max_col_test_1 = np.amax(img[config.alpha_y1,base_x:base_x+config.alpha_width])
        max_col_test_2 = np.amax(img[config.alpha_y2,base_x:base_x+config.alpha_width])

        if max_col_test_1 > max_color or max_col_test_2 > max_color:
            return False

        # Bravo team testing
        base_x = config.bravo_base_x + config.bravo_spacing * player

        max_col_test_1 = np.amax(img[config.bravo_y1,base_x:base_x+config.bravo_width])
        max_col_test_2 = np.amax(img[config.bravo_y2,base_x:base_x+config.bravo_width])

        if max_col_test_1 > max_color or max_col_test_2 > max_color:
            return False

    return True


def get_obj_state(img):
    state_matches = [i for i, state in enumerate(state_list) if matches_state(img, state)]
    if len(state_matches) != 1:
        return NO_OBJ_STATE
    return state_matches[0] - 2

def push_obj_state(state_buffer, new_state):
    state_buffer.append(new_state)
    state_buffer.pop(0)

def obj_state_to_string(assumed_state):
    if assumed_state == 2:
        return 'alpha++'
    elif assumed_state == 1:
        return 'alpha+'
    elif assumed_state == 0:
        return 'neutral'
    elif assumed_state == -1:
        return 'bravo+'
    elif assumed_state == -2:
        return 'bravo++'

class ObjectiveTimer:
    def __init__(self):
        self.objective_time = [0]*5
        self.current_hold_length = 0
        self.longest_hold_alpha = 0
        self.longest_hold_bravo = 0
        self.obj_state_buffer = [NO_OBJ_STATE]*10
        self.last_assumed_state = NO_OBJ_STATE
        self.assumed_state = NO_OBJ_STATE
        self.in_game = False
        self.was_in_game = False
        self.game_time = 0
        self.event_queue = []

        self.prev_game_event_queue = []
        self.prev_game_objective_time = [0]*5
        self.prev_game_longest_hold_alpha = 0
        self.prev_game_longest_hold_bravo = 0

    def update(self, img, dt):
        self.was_in_game = self.in_game
        self.last_assumed_state = self.assumed_state
        true_obj_state = get_obj_state(img)
        self.obj_state_buffer.append(true_obj_state)
        self.obj_state_buffer.pop(0)
        if self.is_game_start():
            self.in_game = True
            self.game_time = 0
            evnt = {'event_type': 'game_start', \
                'game_time_seconds': round(self.game_time, 2), \
                'timestamp': round(time.time(), 2)}
            self.event_queue.append(evnt)
        if self.in_game:
            self.game_time += dt
            # If latest frame gives us a NO_OBJ_STATE, find the most recent objective state in the buffer
            self.assumed_state = next((s for s in reversed(self.obj_state_buffer) if s != NO_OBJ_STATE), NO_OBJ_STATE)
            if self.assumed_state != NO_OBJ_STATE and self.last_assumed_state != NO_OBJ_STATE:
                self.objective_time[self.assumed_state + 2] += dt
                self.current_hold_length += dt
                if self.last_assumed_state != self.assumed_state:
                    evnt = {'event_type': 'objective_change', \
                        'game_time_seconds': round(self.game_time, 2)}
                    evnt['old_objective_state'] = obj_state_to_string(self.last_assumed_state)
                    evnt['new_objective_state'] = obj_state_to_string(self.assumed_state)
                    self.event_queue.append(evnt)
                if np.sign(self.last_assumed_state) != np.sign(self.assumed_state):
                    if np.sign(self.last_assumed_state) > 0:
                        self.longest_hold_alpha = max(self.longest_hold_alpha, self.current_hold_length)
                    elif np.sign(self.last_assumed_state) < 0:
                        self.longest_hold_bravo = max(self.longest_hold_bravo, self.current_hold_length)
                    self.current_hold_length = 0
            if self.is_game_end():
                evnt = {'event_type': 'game_end', \
                    'game_time_seconds': round(self.game_time, 2), \
                    'timestamp': round(time.time(), 2)}
                self.event_queue.append(evnt)
                self.reset()
                
    def is_game_start(self):
        return not self.was_in_game and len([s for s in self.obj_state_buffer[-5:] if s == NEUTRAL_OBJ_STATE]) >= 3

    def is_game_end(self):
        return self.was_in_game and len([s for s in self.obj_state_buffer[-6:] if s == NO_OBJ_STATE]) >= 5

    def is_objective_control_change(self):
        return self.last_assumed_state != NO_OBJ_STATE \
        and self.assumed_state != NO_OBJ_STATE \
                    and np.sign(self.last_assumed_state) != np.sign(self.assumed_state)

    def reset(self):
        self.prev_game_event_queue = self.event_queue
        self.prev_game_longest_hold_alpha = round(self.longest_hold_alpha, 2)
        self.prev_game_longest_hold_bravo = round(self.longest_hold_bravo, 2)
        self.prev_game_objective_time = [round(x, 2) for x in self.objective_time]

        self.objective_time = [0]*5
        self.obj_state_buffer = [NO_OBJ_STATE]*10
        self.last_assumed_state = NO_OBJ_STATE
        self.assumed_state = NO_OBJ_STATE
        self.in_game = False
        self.longest_hold_alpha = 0
        self.longest_hold_bravo = 0
