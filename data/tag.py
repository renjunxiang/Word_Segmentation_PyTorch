START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"
tag_to_ix = {
    PAD_TAG: 0,
    START_TAG: 1,
    STOP_TAG: 2,
    'B': 3, 'M': 4, 'E': 5,
    'S': 6
}