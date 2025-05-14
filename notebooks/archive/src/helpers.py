import pandas as pd

class RuntimeStorage:
    def __init__(self):
        self.storage = list()

    def add_entry(self, original_id, original_input, true_label, model_output, task):
        tmp = dict()
        tmp['originalId'] = None
        tmp['originalInput'] = None
        tmp['trueLabel'] = None
        tmp['modelOutput'] = None
        tmp['task'] = None
        self.storage.append(tmp)

    def export_to_csv(self, filename):
        tmp_df = pd.DataFrame(self.storage)
        tmp_df.to_csv(filename, index=False)
