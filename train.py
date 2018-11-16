import argparse
import os
import time
from collections import deque

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model
from config import Config, TrainingConfig


def _train(path_to_data_dir: str, path_to_checkpoints_dir: str):
    os.makedirs(path_to_checkpoints_dir, exist_ok=True)

    # TODO: CODE BEGIN
    # raise NotImplementedError
    dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TRAIN)
    dataloader = DataLoader(dataset, batch_size=TrainingConfig.Batch_Size, shuffle=True)
    # TODO: CODE END

    # TODO: CODE BEGIN
    # raise NotImplementedError
    model = Model()
    if Config.Device == 'gpu':
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.Learning_Rate)
    # TODO: CODE END

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False

    print('Start training')

    while not should_stop:
        for batch_idx, (images, labels) in enumerate(dataloader):
            if Config.Device == 'gpu':
                images = images.cuda()
                labels = labels.cuda()

            # TODO: CODE BEGIN
            # raise NotImplementedError
            logits = model.train().forward(images)
            loss = model.loss(logits, labels)
            # TODO: CODE END

            # TODO: CODE BEGIN
            # raise NotImplementedError
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # TODO: CODE END

            losses.append(loss.item())
            step += 1

            if step % TrainingConfig.EveryStepsToCheckLoss == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = TrainingConfig.EveryStepsToCheckLoss / elapsed_time
                avg_loss = sum(losses) / len(losses)
                print(f'[Step {step}] Avg. Loss = {avg_loss:.6f} ({steps_per_sec:.2f} steps/sec)')

            if step % TrainingConfig.EveryStepsToSnapshot == 0:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                print(f'Model saved to {path_to_checkpoint}')

            if step % TrainingConfig.StepToDecay == 0:
                optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.Learning_Rate/2)
                print(f'Learning rate changed to {TrainingConfig.Learning_Rate/2}')

            if step == TrainingConfig.StepToFinish:
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
