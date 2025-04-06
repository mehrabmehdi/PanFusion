from .PanoDataset import PanoDataset, PanoDataModule
from .prompts import hex_prompt_pairs

class DemoDataset(PanoDataset):
    def load_split(self, mode):
        # with open(self.data_dir) as f:
        #     data = f.readlines()
        data = [{'pano_prompt': d[1].strip(), 'pano_id': d[0]} for d in hex_prompt_pairs]
        return data

    def get_data(self, idx):
        data = self.data[idx].copy()
        # data['pano_id'] = f"{idx:06d}"
        return data


class Demo(PanoDataModule):
    def __init__(
            self,
            data_dir: str = 'data/Demo/captions.txt',
            *args,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.dataset_cls = DemoDataset
