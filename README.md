# NeuroRehack-Project-1

low-cost audio-visual feedback system for neuro-rehabilitation exercise verfication and report generation.

# Usage

To process the video file of an action, so as to store salient data from openpose as json, you'll need to provide the action folder's name as an argument.

  ```
  python create_json_recordings.py --folder=action1
  ```

  The expected folder structure is:
  - video_recordings
    - action1
      - recording.mp4
    - action2
    ...

To compare a user's current action (from webcam) to that of a recorded one, you'll need to provide the action folder's name as an argument. Note that the json data of this action (from the above step) needs to already exist.

```
python compare.py --folder=action1
```

The expected folder structure is:
- json_recordings
  - action1
    - recording_timestamp.json
  - action2
  ...


## Relevant Links
### Recommended tool for recording and editing action videos:-
https://webcamera.io/

### Datasets:-
https://docs.google.com/document/d/1KZ_6ZwL2hCQgLg7qI6_QH7wH0Z4PDG3t8uKVXPLAaFY/edit?usp=sharing

### Common Problems Faced while installing OpenPose:-
https://docs.google.com/document/d/1KZ_6ZwL2hCQgLg7qI6_QH7wH0Z4PDG3t8uKVXPLAaFY/edit?usp=sharing
