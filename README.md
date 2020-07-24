# Splatoon 2 Spectator Mode Parser
## Installation
1. Update OBS. Ensure you are running a more recent release
2. Install [OBS Virtual Cam](https://github.com/Fenrirthviti/obs-virtual-cam/releases)
3. When prompted, install at least 1 virtual camera
4. Install [Python 3.6+](https://www.python.org/downloads/)
5. Head into the project root directory and run the following command.
```
bash ocr-data/install-dependencies.sh
```
6. If you do not have bash installed, you can manually run the commands within the file.

## Setup
This program uses CV2 VideoCapture. Your computer's webcams and virtual cameras are accessed using an index. If you have a built-in webcam, it is typically index **0**. In general, the index of your first installed virtual camera is **1**, but this may differ between systems. Follow the below instructions to configure your virtual camera.

1. If you have a built-in webcam, run the following command. The input parameter represents the webcam/virtual camera to show. It should up a window displaying its video feed. Press escape to exit.
```
python ocr-data/show_webcam.py 0
```
2. Open OBS. Go to Settings -> Video -> Base Canvas Resolution and ensure that it is 1920x1080. 

3. Create a scene containing a 1920x1080 feed from your capture card. Ensure there are no extraneous sources blocking the video feed on the scene.

4. In the OBS header bar, go to Tools -> VirtualCam

5. A window should pop up with a **Target Camera** in addition to a **Start** and **Stop** button. Your *Target Camera* should look like 'OBS-Camera' if VirtualCam was installed correctly. Hit **Start** to begin redirecting your OBS video to a virtual camera.

6. Rerun the same command, but with a different input camera. If a window pops up showing your OBS video, you have found the correct index for your virtual camera. Again, press escape to exit. 

```
python ocr-data/show_webcam.py 1
```

7. If no index between 1 and 10 seems to match your video, then an issue may have occurred during installation. Uninstalling and reinstalling OBS Virtual Cam may fix your issue. If not, you can contact me and I'd be happy to help set it up.

8. After finding your virtual cam index, you can finally set up your spectator cam parser.

9. Set up your overhead spectator camera in Splatoon 2. For best results, disable screen burn-in reduction (the screen dimming after 5 minutes) and all notifications in system settings. Don't move the camera or change perspectives while parsing.

10. Run the following command with the virtual cam index found earlier to start up the spectator cam parser. 

```
python ocr-data/parse_spectator_stream.py ${index}
```

11. You may see some logs related to 'true_divide', or 'misaligned data'. These errors can be ignored. As long as the script continues running, it will work. Yay coding :)

12. Results will be output in the project root directory under the **/events/** folder.

13. Happy Splatting!