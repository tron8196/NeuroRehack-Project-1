# NeuroRehack-Project-1

low-cost visual feedback system for neuro-rehabilitation exercise verfication and report generation.

# Usage

Run the python program with 'record' or 'compare' argument. In record mode, you'll need to provide the action's folder name as well. By default it is set to 'action1'.

```
python op_webcam.py record --folder=action2
```

The expected folder structure is:
- recordings
  - action1
    - recording_timestamp.json
    ...
  - action2


In compare mode, you'll need to provide a previous recording's json file path to the --data argument.

  ```
  python op_webcam.py compare --data=./recordings/action1/recording_timestamp.json
  ```

## Relevant Links
### Datasets :- https://docs.google.com/document/d/1KZ_6ZwL2hCQgLg7qI6_QH7wH0Z4PDG3t8uKVXPLAaFY/edit?usp=sharing
### Common Problems Faced while installing OpenPose :- https://docs.google.com/document/d/1KZ_6ZwL2hCQgLg7qI6_QH7wH0Z4PDG3t8uKVXPLAaFY/edit?usp=sharing
