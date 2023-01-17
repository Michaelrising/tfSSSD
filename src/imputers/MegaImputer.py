from MegaModel import Mega


class MegaImputer:
    def __init__(self, model_path, log_path, config_path):
        self.model_path=model_path
        self.log_path = log_path
        self.config_path = config_path
        self.model = None



    def train(self,
              data,
              masking='rm',
              infer_flag=False):

        self.model = Mega(num_tokens=256,
                          dim=512,
                          depth=8)




