import pandas as pd

class RuntimeStorage:
    def __init__(self):
        self.storage = list()

    def add_entry(self, original_id, model_output, task):
        tmp = dict()
        tmp['originalId'] = None
        tmp['modelOutput'] = None
        tmp['task'] = None
        self.storage.append(tmp)
        pass

    def export_to_csv(self, filename):
        tmp_df = pd.DataFrame(self.storage)
        tmp_df.to_csv(filename, index=False)
