import numpy as np
import pickle as pkl
import time
from mindspore.dataset import GeneratorDataset, RandomSampler


class HMERDataset:
    def __init__(self, params, image_path, label_path, words, is_train=True):
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = pkl.load(f)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        name = name.split('.')[0] if name.endswith('jpg') else name

        image = self.images[name]
        image = np.array(255 - image, dtype=np.float32) / 255

        init_images = np.zeros((self.params['image_shape']['height'], self.params['image_shape']['width']), dtype=np.float32)
        init_image_masks = np.zeros((self.params['image_shape']['height'], self.params['image_shape']['width']), dtype=np.int32)

        init_images[:image.shape[0], :image.shape[1]] = image
        init_images = np.expand_dims(init_images, 0)
        init_image_masks[:image.shape[0], :image.shape[1]] = np.ones(image.shape, dtype=np.int32)
        init_image_masks = np.expand_dims(init_image_masks, 0)

        labels.append('eos')
        words = self.words.encode(labels)
        words = np.array(words, dtype=np.int32)

        init_words = np.zeros(self.params['label_len'], dtype=np.int32)
        init_words_masks = np.zeros(self.params['label_len'], dtype=np.int32)
        init_words[:len(words)] = words
        init_words_masks[:len(words)] = np.ones(len(words), dtype=np.int32)
        return init_images, init_image_masks, init_words, init_words_masks


def get_crohme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"训练数据路径 images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"验证数据路径 images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
    eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

    train_sampler = RandomSampler(replacement=False)
    eval_sampler = RandomSampler(replacement=False)

    train_loader = GeneratorDataset(train_dataset, sampler=train_sampler, column_names=['images', 'image_masks', 'words', 'words_masks']).batch(params['batch_size'])
    eval_loader = GeneratorDataset(eval_dataset, sampler=eval_sampler, column_names=['images', 'image_masks', 'words', 'words_masks']).batch(1)

    # print(f'train dataset: {train_dataset} train steps: {len(train_loader)} '
    #       f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader


class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label


if __name__ == '__main__':
    from utils import load_config
    config_file = 'config.yaml'
    params = load_config(config_file)
    image_path = './datasets/CROHME/train_images.pkl'
    label_path = './datasets/CROHME/train_labels.txt'
    word_path = './datasets/CROHME/words_dict.txt'
    words = Words(word_path)
    raw = HMERDataset(params, image_path, label_path, words)
    dataset = GeneratorDataset(source=raw, column_names=["data", "label", "label_mask"])
    x = next(iter(get_crohme_dataset(params)[0]))