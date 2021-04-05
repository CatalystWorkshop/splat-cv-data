import numpy as np

# Dead squids have only 3 colors. 0s, 95s, and 200s. 
# May be some intermediate grey artifacts - checks if certain proportion match these colors
def is_dead_squid_colors(img):
    allow_error = 10
    col1 = 0
    col2 = 95
    col3 = 150
    col3 = 200
    a1 = (img < col1 + allow_error).all(axis=-1)
    a2 = ((img > col2 - allow_error) & (img < col2 + allow_error)).all(axis=-1)
    a3 = ((img > col3 - allow_error) & (img < col3 + allow_error)).all(axis=-1)

    bitwise_or = a1 | a2 | a3
    okay_pixels = np.count_nonzero(bitwise_or)
    total_pixels = bitwise_or.size
    return okay_pixels / total_pixels >= 0.65


def get_player_alive_state(img, ui_spacing):
    is_alive_state = [False]*8
    for player in range(4):
        #Alpha team testing
        base_x = ui_spacing.alpha_base_x + ui_spacing.alpha_spacing * player
        base_y = ui_spacing.alpha_y2
        width = ui_spacing.alpha_width
        is_alive_state[player] = not is_dead_squid_colors(img[base_y:base_y + 2 * width,base_x:base_x + width])

        # Bravo team testing
        base_x = ui_spacing.bravo_base_x + ui_spacing.bravo_spacing * player
        base_y = ui_spacing.bravo_y2
        width = ui_spacing.bravo_width
        is_alive_state[player + 4] = not is_dead_squid_colors(img[base_y:base_y + 2 * width,base_x:base_x + width])

    return is_alive_state