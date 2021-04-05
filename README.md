# Splatoon 2 Spectator Mode Parser
## Installation
1. Update OBS. Ensure you are running a more recent release
2. Install [OBS Virtual Cam](https://github.com/Fenrirthviti/obs-virtual-cam/releases)
3. When prompted, install at least 1 virtual camera
4. Install [Python 3.6+](https://www.python.org/downloads/)
5. Head into the project root directory and run the following command.
```
pip install -r requirements.txt
```

## Setup
This program uses CV2 VideoCapture. Your computer's webcams and virtual cameras are accessed using an index. If you have a built-in webcam, it is typically index **0**. In general, the index of your first installed virtual camera is **1**, but this may differ between systems. Follow the below instructions to configure your virtual camera.

1. If you have a built-in webcam, run the following command. The input parameter represents the webcam/virtual camera to show. It should up a window displaying its video feed. Press escape to exit.
```
python ocr-data/show_webcam.py 0
```
2. Open OBS. Go to Settings -> Video -> Base Canvas Resolution and ensure that it is 1920x1080. 

3. Create a scene containing a 1920x1080 feed from your capture card. Ensure there are no extraneous sources blocking the video feed on the scene.

4. In the OBS header bar, go to Tools -> VirtualCam

5. A window should pop up with a **Target Camera** in addition to a **Start** and **Stop** button. Your *Target Camera* should look like 'OBS-Camera' if VirtualCam was installed correctly. Hit **Start** to begin redirecting your OBS video to a virtual camera. When you are done using the spectator cam parser, hit **Stop**

6. Rerun the same command, but with a different input camera. If a window pops up showing your OBS video, you have found the correct index for your virtual camera. Again, press escape to exit. 

```
python ocr-data/show_webcam.py 1
```

7. If no index between 1 and 10 seems to match your video, then an issue may have occurred during installation. Uninstalling and reinstalling OBS Virtual Cam may fix your issue. If not, you can contact me and I'd be happy to help set it up.

8. After finding your virtual cam index, you can finally set up your spectator cam parser.

9. Set up your overhead spectator camera in Splatoon 2. For best results, disable screen burn-in reduction (the screen dimming after 5 minutes) and all notifications in Switch system settings. Don't move the camera or change perspectives while parsing.

10. Run the following command with the virtual cam index found earlier to start up the spectator cam parser. 

```
python ocr-data/parse_spectator_stream.py ${index}
```

11. You may see some logs related to 'true_divide', or 'misaligned data'. These errors can be ignored. As long as the script continues running, it will work. Yay coding :)

12. Results will be output in the project root directory under the **/events/** folder.

13. Use Ctrl + C to exit the program.

14. Happy Splatting!


## Understanding the Output

### Example Output (Abridged for simplicity)
```
{
    "event_source": "CV",
    "timestamp": 1595529832.9844382,
    "event_type": "results",
    "event_data": {
        "game_mode": "Splat Zones",
        "stage": "Shellendorf Institute",
        "winner": "bravo",
        "game_events": [
            {
                "event_type": "game_start",
                "game_time_seconds": 0,
                "timestamp": 1595529890.28
            },
            {
                "event_type": "player_splatted",
                "player_number": 3,
                "team": "alpha",
                "game_time_seconds": 9.88
            },
            {
                "event_type": "player_splatted",
                "player_number": 1,
                "team": "alpha",
                "game_time_seconds": 12.34
            },
            {
                "event_type": "player_splatted",
                "player_number": 5,
                "team": "bravo",
                "game_time_seconds": 12.34
            },
            {
                "event_type": "objective_change",
                "game_time_seconds": 16.09,
                "old_objective_state": "neutral",
                "new_objective_state": "bravo++"
            },
            {
                "event_type": "player_splatted",
                "player_number": 4,
                "team": "alpha",
                "game_time_seconds": 16.83
            },
            {
                "event_type": "player_respawn",
                "player_number": 3,
                "team": "alpha",
                "game_time_seconds": 18.39
            }
            {
                "event_type": "game_end",
                "game_time_seconds": 321.16,
                "timestamp": 1595530211.42
            }
        ],
        "objective_time": [
            49.47,
            74.59,
            64.43,
            86.17,
            46.31
        ],
        "alpha_team": {
            "players": [
                {
                    "weapon": "Inkbrush Nouveau",
                    "ka_count": 14,
                    "special": "Baller",
                    "special_count": 5
                },
                {
                    "weapon": "Firefin Splatterscope",
                    "ka_count": 11,
                    "special": "Suction-Bomb Launcher",
                    "special_count": 4
                },
                {
                    "weapon": "Kensa Splattershot Pro",
                    "ka_count": 8,
                    "special": "Booyah Bomb",
                    "special_count": 4
                },
                {
                    "weapon": "Sorella Brella",
                    "ka_count": 8,
                    "special": "Splat-Bomb Launcher",
                    "special_count": 3
                }
            ],
            "longest_hold_alpha": 28.99
        },
        "bravo_team": {
            "players": [
                {
                    "weapon": "Dualie Squelchers",
                    "ka_count": 11,
                    "special": "Tenta Missiles",
                    "special_count": 4
                },
                {
                    "weapon": "Enperry Splat Dualies",
                    "ka_count": 10,
                    "special": "Inkjet",
                    "special_count": 3
                },
                {
                    "weapon": "Dark Tetra Dualies",
                    "ka_count": 9,
                    "special": "Splashdown",
                    "special_count": 3
                },
                {
                    "weapon": "Firefin Splatterscope",
                    "ka_count": 5,
                    "special": "Suction-Bomb Launcher",
                    "special_count": 5
                }
            ],
            "longest_hold_bravo": 31.07
        }
    }
}
```

### Event Types
1. Game Start/Game End

    These should be self-explanatory. The game time starts at `game_time_seconds=0` and ends at `game_time_seconds=321.16` in this example.
    ```{
            "event_type": "game_start",
            "game_time_seconds": 0,
            "timestamp": 1595529890.28
        },
        {
            "event_type": "game_end",
            "game_time_seconds": 321.16,
            "timestamp": 1595530211.42
        }
    ```
2. Player Splatted/Player Respawn

    These events show the player number that died or respawned, their team, and the time it occurred.
    By convention, Players 1-4 are Alpha Team and Players 5-8 are Bravo Team.
    Player number order follows their order in the PB lobby, from top to bottom, alpha team then bravo team.
    ```
        {
            "event_type": "player_splatted",
            "player_number": 3,
            "team": "alpha",
            "game_time_seconds": 9.88
        },
        {
            "event_type": "player_respawn",
            "player_number": 3,
            "team": "alpha",
            "game_time_seconds": 18.39
        }
    ```

3. Objective Change

    The objective has 5 states:
    - bravo++: Bravo has control and is scoring points
    - bravo+: Bravo has control but is not scoring points (Ex, eating away at penalty)
    - neutral: Neither team has objective control.
    - alpha+: Alpha has control but is not scoring points (Ex, eating away at penalty)
    - alpha++: Alpha has control and is scoring points

    Both the previous state if the objective and the new state are logged.

    ```
        {
            "event_type": "objective_change",
            "game_time_seconds": 16.09,
            "old_objective_state": "neutral",
            "new_objective_state": "bravo++"
        }
    ```

4. Objective Time

    Objective time is returned in an array representing the 5 states detailed above.

    ```
        "objective_time": [
            49.47, (bravo++)
            74.59, (bravo+)
            64.43, (neutral)
            86.17, (alpha+)
            46.31 (alpha++)
        ]
    ```

5. Match Results

    The match results are fairly self explanatory. They do not include deaths, as that info does not appear on the game screen.

    These players are not particularly associated with a certain player number, as the order players appear on this screen does not match the lobby order.

    (Code that does this is WIP and requires identifying player names)

    `longest_hold` refers to the longest period of time that a team held the objective (or it was neutral) without the enemy team gaining control.
