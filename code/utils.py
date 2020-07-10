def tag2idx(data):
    tag_values = list(set(data["Tag"].values))
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    return tag2idx