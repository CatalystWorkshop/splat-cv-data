#!/usr/bin/env python

import asyncio
import websockets
from collections import namedtuple
import json
import base64
import io
import PySimpleGUI as sg             
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageTk
# sg.change_look_and_feel('DarkAmber')


PossibleResults = namedtuple('PossibleResults', 'weapon special abilities possibleStats')
PossibleStats = namedtuple('PossibleStats', 'ka_count special_count')
Result = namedtuple('Result', 'player_name weapon special abilities ka_count special_count')
ResultsList = namedtuple('ResultsList', 'winner data')
def decode_image(img_data):
	return Image.open(io.BytesIO(base64.b64decode(img_data.encode('utf-8'))))

def get_img_data(img, maxsize=(640, 720), first=False):
    """Generate image data using PIL
    """
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

def exist_conflicts(assoc_results):
	return not all([len(res.possibleStats) == 1 for res in assoc_results])

def resolve_associated_conflicts(player_list, res_img, assoc_results):
	# if not exist_conflicts(assoc_results):
	# 	return ResultsList(assoc_results.winner, [Result(player, res.weapon, res.special, res.abilities, res.possibleStats[0].ka_count, res.possibleStats[0].special_count) \
	# 	for player, res in zip(player_list, assoc_results.data)])
	# layout = [[sg.Text(player), \
	# sg.Text(str(res.possibleStats[0].ka_count) + "," + str(res.possibleStats[0].special_count)) if len(res.possibleStats) == 1 \
	# else sg.Drop([str(posRes.ka_count) + "," + str(posRes.special_count) for posRes in res.possibleStats], key=('stats'+str(i)))] \
	# for i,(player,res) in enumerate(zip(player_list, assoc_results))]
	# layout.append([sg.Submit()])
	# layout.insert(0, [sg.Image(data=get_img_data(res_img, first=True))])

	# window = sg.Window('Window Title').Layout(layout)
	# event, values = window.Read(timeout=100)
	# fully_assoc_res = [None] * 8
	# if event == 'Submit':
	# 	for i in range(8):
	# 		ka_count = assoc_results[i].possibleStats[0].ka_count
	# 		special_count = assoc_results[i].possibleStats[0].special_count

	# 		key = 'stats' + str(i)
	# 		if key in values:
	# 			val = values['stats' + str(i)].split(',')
	# 			if len(val) == 2:
	# 				ka_count = int(val[0])
	# 				special_count = int(val[1])

	# 		fully_assoc_res[i] = Result(player_list[i], assoc_results[i].weapon, assoc_results[i].special, \
	# 		assoc_results[i].abilities, ka_count, special_count)
	# print("2", flush=True)
	# return fully_assoc_res
	print('hi')

def json_map_to_possible_results(data):
	return [PossibleResults(player['weapon'], player['special'], (player['headgear'], player['clothing'], player['shoes'])\
		,[PossibleStats(posRes['ka_count'], posRes['special_count']) for posRes in player['possibleResults']]) for player in data['players']]

async def echo(websocket, path):
	async for message in websocket:
		print('Data received', flush=True)
		data = json.loads(message)
		res_img = decode_image(data['resultsImage'])
		assoc_results = json_map_to_possible_results(data)
		player_list = ['Snek', 'Ross', 'Nick', '4D', 'Power', 'Astral', 'Prod', 'Keen']
		#await asyncio.get_event_loop().run_in_executor(None, partial(resolve_associated_conflicts, player_list, res_img, assoc_results))
		# if not exist_conflicts(assoc_results):
		# 	print('test', flush=True)
		# resolve_associated_conflicts(player_list, res_img, assoc_results)
		# else:
		# 	print('test2', flush=True)
		# 	data_queue.put((player_list, res_img, assoc_results))

# def resolve_conflict_loop():
# 	while True:
# 		(player_list,res_img,assoc_results) = await data_queue.get()
# 		print(assoc_results, flush=True)
# 		resolve_associated_conflicts(player_list, res_img, assoc_results)

id = 0
data_queue = queue.Queue()

def main():
	start_server = websockets.serve(echo, "localhost", 8765)

	loop = asyncio.get_event_loop()
	loop.run_until_complete(start_server)
	# loop.run_until_complete(resolve_conflict_loop())
	loop.run_forever()
	# asyncio.ensure_future(start_server)
	# loop.run_forever()


if __name__ == '__main__':
	main()