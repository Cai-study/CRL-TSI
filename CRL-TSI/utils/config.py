class Config:
    def __init__(self,task):
        if "mmwhs" in task:
            self.save_dir = './MMWHS_data'
            self.patch_size = (128, 128, 128)
            self.num_cls = 5
            self.num_channels = 1
            self.n_filters = 32
            self.batch_size = 2
        elif "abdominal" in task:
            self.save_dir = './abdominal_data'
            self.patch_size = (128, 128, 128)
            self.num_cls = 5
            self.num_channels = 1
            self.n_filters = 32
            self.batch_size = 2

        else:
            raise NameError("Please provide correct task name, see config.py")


