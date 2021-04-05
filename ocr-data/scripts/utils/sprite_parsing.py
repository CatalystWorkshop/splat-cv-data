from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize


class Spritesheet:
    def __init__(self, sheet, num_sprites, num_cols, sprite_width, sprite_height, sprite_names):
        self.sheet = sheet
        self.num_sprites = num_sprites
        self.num_cols = num_cols
        self.sprite_width = sprite_width
        self.sprite_height = sprite_height
        self.sprite_names = sprite_names
        self.cached_sprites = {}

    def get_sprite(self, idx):
        if idx < 0 or idx >= self.num_sprites:
            print("error, invalid sprite number")
            return ("", [])
        if idx not in self.cached_sprites.keys():
            x = self.sprite_width * (idx % self.num_cols)
            y = self.sprite_height * int(idx / self.num_cols)
            self.cached_sprites[idx] = (self.sprite_names[idx], self.sheet[y:y+self.sprite_height, x:x+self.sprite_width])
        return self.cached_sprites[idx]

# Takes in an RGB image
def find_most_similar(in_img, spritesheet, permit_list=None):
    max_score = 0
    max_img = ''
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