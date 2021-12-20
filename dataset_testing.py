from training.dataset import MeshRenderingDataset

if __name__ == "__main__":
    dataset = MeshRenderingDataset(
        path='../data/ffhq_mesh'
    )
    dataset.get_obj_fnames()