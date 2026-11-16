"""Shared constants for LeapGestureDB.

Feature layout of one frame (26 values, as exported by the Leap Motion SDK
in LeapGestureDB):
    [0]      hand id / type            [1]      finger count
    [2:5]    hand direction (unit)     [5:8]    palm position (mm)
    [8:11]   palm normal (unit)        [11:14]  thumb tip (mm)
    [14:17]  index tip (mm)            [17:20]  middle tip (mm)
    [20:23]  ring tip (mm)             [23:26]  pinky tip (mm)
"""

GESTURE_MAPPING = {f"G{i}": i - 1 for i in range(1, 12)}
GESTURE_NAMES = [
    "Click", "Left rotation (CW)", "Right rotation (CCW)",
    "Increase contrast", "Decrease contrast", "Zoom in", "Zoom out",
    "Move left", "Move right", "Previous", "Next",
]
FEATURE_NAMES = [
    "HandID", "FingerCount",
    "HandDirection_X", "HandDirection_Y", "HandDirection_Z",
    "PalmPosition_X", "PalmPosition_Y", "PalmPosition_Z",
    "PalmNormal_X", "PalmNormal_Y", "PalmNormal_Z",
    "ThumbTip_X", "ThumbTip_Y", "ThumbTip_Z",
    "IndexTip_X", "IndexTip_Y", "IndexTip_Z",
    "MiddleTip_X", "MiddleTip_Y", "MiddleTip_Z",
    "RingTip_X", "RingTip_Y", "RingTip_Z",
    "PinkyTip_X", "PinkyTip_Y", "PinkyTip_Z",
]
NUM_CLASSES = 11
INPUT_DIM = 26
SEQ_LENGTH = 100

IDX_HAND_ID = 0
IDX_FINGERS = 1
SL_DIR = slice(2, 5)
SL_PALM = slice(5, 8)
SL_NORMAL = slice(8, 11)
SL_TIPS = slice(11, 26)          # 5 tips x 3
POSITION_IDX = list(range(5, 8)) + list(range(11, 26))   # all positional (mm) features
UNIT_VECTOR_IDX = list(range(2, 5)) + list(range(8, 11))

# Feature groups used to aggregate attributions (reported per group, not per
# single coordinate, because single coordinates are strongly collinear).
FEATURE_GROUPS = {
    "Finger count": [1],
    "Hand direction": [2, 3, 4],
    "Palm position": [5, 6, 7],
    "Palm normal": [8, 9, 10],
    "Thumb tip": [11, 12, 13],
    "Index tip": [14, 15, 16],
    "Middle tip": [17, 18, 19],
    "Ring tip": [20, 21, 22],
    "Pinky tip": [23, 24, 25],
}
