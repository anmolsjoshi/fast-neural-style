import argparse
import os
import sys
import time
import re

import numpy as np
import random
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer

import utils
from transformer_net import TransformerNet
from vgg import Vgg16

STYLIZED_IMG_FNAME = 'stylized_sample_epoch_{:04d}.png'


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def check_manual_seed(seed):

    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.image_size)
    style = style_transform(style)
    style = style.repeat(1, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    test_image = utils.load_image(args.test_image, size=args.image_size)
    test_image = style_transform(test_image)
    test_image = test_image.to(device)

    def step(engine, batch):

        x, _ = batch
        x = x.to(device)

        n_batch = len(x)

        transformer.zero_grad()

        y = transformer(x)

        x = utils.normalize_batch(x)
        y = utils.normalize_batch(y)

        features_x = vgg(x)
        features_y = vgg(y)

        content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

        style_loss = 0.
        for ft_y, gm_s in zip(features_y, gram_style):
            gm_y = utils.gram_matrix(ft_y)
            style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

        tv_loss = args.tv_weight * (mse_loss(y[:, :, :, :-1], y[:, :, :, 1:]) +
                                    mse_loss(y[:, :, :-1, :], y[:, :, 1:, :]))

        total_loss = content_loss + style_loss + tv_loss
        total_loss.backward()
        optimizer.step()


        return {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'tv_loss': tv_loss.item(),
            'total_loss': total_loss.item()
        }

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(args.ckpt_dir, 'ckpt_epoch_',
                                         save_interval=args.checkpoint_interval,
                                         n_saved=10, require_empty=False)
    timer = Timer(average=True)

    print (trainer)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % args.log_interval == 0:
            message = 'Epoch[{epoch}/{max_epoch}] Iteration[{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                                  max_epoch=args.max_epochs,
                                                                  i=(engine.state.iteration % len(train_loader)),
                                                                  max_i=len(train_loader))
            for name, value in engine.state.output.items():
                message += ' | {name}: {value:.4f}'.format(name=name, value=value)

            print(message)

    @trainer.on(Events.EPOCH_COMPLETED)
    def stylize_image(engine):
        stylized_test_image = transformer(test_image)
        path = os.path.join(args.stylized_dir, STYLIZED_IMG_FNAME.format(engine.state.epoch))
        utils.save_image(path, stylized_test_image)

    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                              handler=checkpoint_handler, to_save={'net': transformer})

    trainer.run(train_loader, max_epochs=1)

def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        output = style_model(content_image).cpu()

    utils.save_image(args.output_image, output[0])

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch_size", type=int, default=32,
                                  help="batch size for training, default is 32")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style_image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    #train_arg_parser.add_argument("--save_model_dir", type=str, required=True,
    #                              help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--ckpt_dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image_size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    #train_arg_parser.add_argument("--style-size", type=int, default=None,
    #                              help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True, default=1,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content_weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style_weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--tv_weight", type=float, default=1e3,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-4,
                                  help="learning rate, default is 1e-4")
    train_arg_parser.add_argument("--log_interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint_interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")


    args = main_arg_parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()


