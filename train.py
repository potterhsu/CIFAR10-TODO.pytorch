import argparse
import os
import time
from collections import deque

import torch
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model


def _train(path_to_data_dir: str, path_to_checkpoints_dir: str):
    os.makedirs(path_to_checkpoints_dir, exist_ok=True)

    # TODO: CODE BEGIN
    raise NotImplementedError
    # dataset = XXX
    # dataloader = XXX
    # TODO: CODE END

    # TODO: CODE BEGIN
    raise NotImplementedError
    # model = XXX
    # optimizer = XXX
    # TODO: CODE END

    num_steps_to_display = 20
    num_steps_to_snapshot = 1000
    num_steps_to_finish = 10000

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False

    print('Start training')

    while not should_stop:
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.cuda()
            labels = labels.cuda()

            # TODO: CODE BEGIN
            raise NotImplementedError
            # logits = XXX
            # loss = XXX
            # TODO: CODE END

            # TODO: CODE BEGIN
            raise NotImplementedError
            # optimizer.XXX
            # loss.XXX
            # optimizer.XXX
            # TODO: CODE END

            losses.append(loss.item())
            step += 1

            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                avg_loss = sum(losses) / len(losses)
                print(f'[Step {step}] Avg. Loss = {avg_loss:.6f} ({steps_per_sec:.2f} steps/sec)')

            if step % num_steps_to_snapshot == 0:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                print(f'Model saved to {path_to_checkpoint}')

            if step == num_steps_to_finish:
                should_stop = True
                break

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
        args = parser.parse_args()

        path_to_data_dir = args.data_dir
        path_to_checkpoints_dir = args.checkpoints_dir

        _train(path_to_data_dir, path_to_checkpoints_dir)

    main()
