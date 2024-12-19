DEFAULT_ST_GRAPH_TOKEN = "<st_graph>"
DEFAULT_ST_GRAPH_PATCH_TOKEN = "<st_patch>"
DEFAULT_ST_GRAPH_START_TOKEN = "<st_start>"
DEFAULT_ST_GRAPH_END_TOKEN = "<st_end>"
n_time_slices = 20
n_time_steps = 3
timestamp = 10

PROMPTS = {}

PROMPTS[
    "node_classification"
] = """-Role-
You are an expert basketball possession analyst who specializes in predicting ball possession using player tracking data.

-Task-
Predict which player will have ball possession in the next few time steps based on current and historical player positions and movement patterns.

-Input-
Given a Graph Transformer encoded graph with {n_time_slices} historical consecutive time slices: {DEFAULT_ST_GRAPH_TOKEN}, with player node information on the field described as {st_player_info}.

-Question-
Starting from the last time slice, who is most likely to have possession in the next {n_time_steps} time slices? Arrange in order from most likely to least likely, in the format "<player_id>" and separated by commas. Please answer strictly according to the answer template. Analyze the given time and player information thoroughly to generate the prediction. Follow a step-by-step approach to avoid incorrect associations.

-Output-
Based on the provided information, <player_id>, <player_id>, <player_id> ... <player_id>.
"""

PROMPTS[
    "link_prediction"
] = """-Role-
You are an expert basketball passing analyst who specializes in predicting ball movement and passing patterns between players using tracking data.

-Task-
Analyze the current game situation and predict whether the ball handler will make a pass, and if so, identify the most likely pass recipients based on player positions, movement patterns, and historical passing tendencies.

-Input-
Given a Graph Transformer encoded graph with {n_time_slices} historical consecutive time slices: {DEFAULT_ST_GRAPH_TOKEN}, with player node information on the field described as {st_player_info}.

-Question-
Starting from the last time slice, will the player with the highest probability of holding the ball in the next time step time slices pass it? If so, to whom? Rank from most likely to least likely, in the format "<yes> or <no>, <player id>", separated by commas. Analyze the time slice based on the provided time and player information, then generate predictions. Think step by step to avoid incorrect assumptions.

-Output-
Based on the provided information, <yes> or <no>, <player_id>, <player_id>, <player_id> ... <player_id>.
"""

PROMPTS[
    "graph_classification"
] = """-Role-
You are an expert basketball shot prediction analyst who specializes in identifying shooting opportunities and player shooting tendencies using tracking data.

-Task-
Analyze the current game situation to predict whether the ball handler will attempt a shot, based on factors such as player positioning, defensive pressure, shot clock, and historical shooting patterns.

-Input-
Given the Graph Transformer encoded graph at {timestamp}: {DEFAULT_ST_GRAPH_TOKEN}, and player node information on the field described as {st_player_info}.

-Question-
Starting from this time slice, will the player with the highest probability of holding the ball in the next {n_time_steps} time slices take a shot? Answer in the format "<yes> or <no>". Analyze the time slice based on the provided time and player information, then generate predictions. Think step by step to avoid incorrect assumptions.

-Output-
Based on the provided information, <yes> or <no>.
"""


PROMPTS[
    "offensive_tactics_description"
] = """
Please provide a detailed description of the offensive tactic based on the given diagram.
"""