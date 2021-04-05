try:
    from PIL import Image
except ImportError:
    import Image
import os
import websockets
import cv2

async def send_to_socket(json_dump):
    uri = "ws://hi:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json_dump)

def save_image(img, now, suffix="", np=False):
    if not os.path.exists("./events"):
        os.makedirs("./events")
    file = './events/' + now.strftime("%Y-%m-%d_%H_%M_%S") + suffix + ".png"
    if np:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil_img = img
    pil_img = pil_img.resize((1280, 720))
    pil_img.save(file)

def log_event(out, now, suffix=""):
    if not os.path.exists("./events"):
        os.makedirs("./events")
    file = './events/' + now.strftime("%Y-%m-%d_%H_%M_%S") + suffix + ".json"
    with open(file, 'w+') as file:
        file.write(out)

def submit_event(json_dump, now):
    log_event(json_dump, now)
    # asyncio.get_event_loop().run_until_complete(send_to_socket(json_dump))