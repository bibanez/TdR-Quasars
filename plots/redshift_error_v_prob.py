from squeze.squeze_common_functions import deserialize, load_json
import matplotlib.pyplot as plt

print("[?] Loading file...")
df = deserialize(load_json("../sample_quasars/candidates_width70_sig6_validation_64plates_sample0.json"))
