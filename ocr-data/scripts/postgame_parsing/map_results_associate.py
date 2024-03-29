import json
import io
import time
import base64
from collections import namedtuple

PossibleResultsList = namedtuple('PossibleResultsList', 'winner data')
PossibleResults = namedtuple('PossibleResults', 'weapon special abilities possibleStats')
PossibleStats = namedtuple('PossibleStats', 'ka_count special_count')

def make_unique(seq):
    seen = set()
    seen_add = seen.add
    for x in seq:
    	seen.add(x)
    return [x for x in seen]

def encode_image(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def associate_players(map_data, match_results):
	map_weaps, abils = zip(*map_data)
	results_weaps, k_a_cnt, spec_cnt, specials = zip(*match_results.data)

	if str(sorted(map_weaps[0:4])) == str(sorted(map_weaps[4:8])):
		winner = match_results.winner
	else:
		winner = 'alpha' if str(sorted(map_weaps[0:4])) == str(sorted(results_weaps[0:4])) \
			and str(sorted(map_weaps[4:8])) == str(sorted(results_weaps[4:8])) else \
		'bravo' if str(sorted(map_weaps[0:4])) == str(sorted(results_weaps[4:8])) \
			and str(sorted(map_weaps[4:8])) == str(sorted(results_weaps[0:4])) else match_results.winner

	player_weapons, player_abils = zip(*map_data)
	alpha_tgts = match_results.data[0:4] if winner == 'alpha' else match_results.data[4:8]
	bravo_tgts = match_results.data[0:4] if winner == 'bravo' else match_results.data[4:8]
	player_results = [None] * 8
	map_specs = [None] * 8
	alphaError = False
	bravoError = False
	for player in range(4):
		# Alpha team
		possibleResults = [PossibleStats(res.ka_count, res.special_count) for res in alpha_tgts if res.weapon == player_weapons[player]]
		num_possible = len(possibleResults)
		if num_possible >= 1:
			player_results[player] = possibleResults
			map_specs[player] = next(res.special for res in alpha_tgts if res.weapon == player_weapons[player])
		else:
			alphaError = True
			print("Unable to determine correct weapon for player " + str(player+1))

		# Bravo team
		possibleResults = [PossibleStats(res.ka_count, res.special_count) for res in bravo_tgts if res.weapon == player_weapons[player + 4]]
		num_possible = len(possibleResults)
		if num_possible >= 1:
			player_results[player + 4] = possibleResults
			map_specs[player + 4] = next(res.special for res in bravo_tgts if res.weapon == player_weapons[player + 4])
		else:
			bravoError = True
			print("Unable to determine correct weapon for player " + str(player+5))
	allAlphaRes = [PossibleStats(res.ka_count, res.special_count) for res in alpha_tgts]
	allBravoRes = [PossibleStats(res.ka_count, res.special_count) for res in bravo_tgts]
	for i in range(4):
		if alphaError:
			player_results[i] = allAlphaRes
			map_specs[i] = "Unknown"
		if bravoError:
			player_results[i + 4] = allBravoRes
			map_specs[i+4] = "Unknown"

	return PossibleResultsList(match_results.winner, [PossibleResults(res[0], res[1], res[2], res[3]) for res in zip(map_weaps, map_specs, abils, [make_unique(res_list) for res_list in player_results])])

def assoc_results_to_json(assoc_results, res_img):
	jsondata = {'eventSource': 'CV', 'timestamp': time.time(), 'eventType': 'results', 'eventData': {'resultsImage': "", 'winner': assoc_results.winner, 'players':[{'weapon': res.weapon, 'special': res.special, \
	'headgear': res.abilities[0], 'clothing': res.abilities[1], 'shoes': res.abilities[2], \
	'possibleResults': [{'ka_count': posRes.ka_count, 'special_count': posRes.special_count} for posRes in res.possibleStats]} for res in assoc_results.data]}}
	return json.dumps(jsondata, indent=4)