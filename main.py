# -*- coding:utf-8 -*-

import os, argparse
import datetime, time
import torch
import torch.nn as nn
import utils, glob
from model import Classifier
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Model Hyperparameters
embedding=utils.load_glove()
vocab_size, embedding_dim = embedding.shape

parser = argparse.ArgumentParser(description="Hyper Parameter Setting")
parser.add_argument("--num_epochs", default=2, type=int, help="Total epoch")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
parser.add_argument("--num_class", default=2, type=int, help="Number of labels list")
parser.add_argument("--vocab_size", default=vocab_size, type=int, help="# of total words ~ 100K")
parser.add_argument("--seq_len", default=500, type=int, help="avg(title+text) ~ 470, max ~ 10000, min ~ 3")
parser.add_argument("--embedding_dim", default=embedding_dim, type=int, help="Dimensionality of text feature")
parser.add_argument("--n_layers", default=1, type=int, help="# of LSTM layer")
parser.add_argument("--lstm_hidden_dim", default=100, type=int, help="LSTM hidden size")

parser.add_argument("--bidirectional", default=True, type=bool, help="Whether to user bidirectional or not")
parser.add_argument("--dropout_keep_prob", default=0.7, type=float, help="Dropout keep probability")
parser.add_argument("--l2_reg_lambda", default=5e-4, type=float, help="weight for L2 regularization")

parser.add_argument("--evaluate_every", default=10, type=int, help="Evaluate model on dev set after this many steps")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="The learning rate (default: 10^-3)")
parser.add_argument("--decay_rate", default=0.9, type=float, help="Rate of decay for learning rate")
args = parser.parse_args()

def clip_gradient(model, clip_value):
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        p.grad.data.clamp_(-clip_value, clip_value)

def calc_acc(pred, true_label, prd_cnt=args.batch_size):
    prob, predicted_ctgr = torch.max(torch.nn.functional.softmax(pred, dim=1), 1)
    match_cnt = (predicted_ctgr == true_label.data).float().sum().item()
    return match_cnt / float(args.batch_size)

def train_model(model, train_iter, epoch):
    total_epoch_loss, total_epoch_acc, steps, best_acc = 0., 0., 0, .7
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'bias' in name: bias_p.append(p)
            else: weight_p.append(p)
    optim = torch.optim.Adam([
        {'params': weight_p, 'weight_decay': args.l2_reg_lambda},
        {'params': bias_p, 'weight_decay':0}
        ], lr=args.learning_rate*(args.decay_rate**epoch))

    if torch.cuda.is_available(): model.cuda()
    model.train()
    for id, text, label in train_iter:
        #if (len(label) != args.batch_size): continue
        label = torch.autograd.Variable(label).long()
        if torch.cuda.is_available():
            text = text.cuda()
            label = label.cuda()

        optim.zero_grad()
        output = model(text)   # [batch_size, sentences]
        loss = loss_ft(output, label)
        acc = calc_acc(output, label)

        loss.backward()
        #clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        total_epoch_loss += loss.item()
        total_epoch_acc += acc

        if  (steps == 1) or (steps % args.evaluate_every  == 0):
            logger.info('\n--- Train Step: %d ---' %steps)
            logger.info('Acc = %.4f, Loss = %.4f' %(acc, loss))
            print('\n--- Train Step: %d ---' %steps)
            print('Acc = %.4f, Loss = %.4f' %(acc, loss))
            writer.add_scalar('Train/Loss', loss, epoch*len(train_iter)+steps)
            writer.add_scalar('Train/Acc', acc*100, epoch*len(train_iter)+steps)
            writer.flush()

        # save checkpoint if acc achieves best acc
        if (acc > best_acc) & (steps % args.evaluate_every  == 0):
            utils.save_ckpt(print_datetime, epoch, model, model_name, acc)
            best_acc = acc

    # averaged loss and acc
    return total_epoch_loss / steps, total_epoch_acc / steps

def eval_model(model, val_iter, epoch=0, tensorboard=False):
    total_epoch_loss, total_epoch_acc, steps = 0., 0., 0
    model.eval()
    # Don't need to gradient flow to optimizer
    with torch.no_grad():
        for id, text, label in val_iter:
            if (len(label) is not args.batch_size): continue
            label = torch.autograd.Variable(label).long()
            if torch.cuda.is_available():
                text = text.cuda()
                label = label.cuda()

            output = model(text)
            loss = loss_ft(output, label)
            acc = calc_acc(output, label)

            steps += 1
            total_epoch_loss += loss.item()
            total_epoch_acc += acc

        if tensorboard:
            writer.add_scalar('Valid/Loss', total_epoch_loss/steps, epoch)
            writer.add_scalar('Valid/Acc', 100*total_epoch_acc/steps, epoch)
            writer.flush()
    return total_epoch_loss/steps, total_epoch_acc/steps

def main():
    #pretrained_embedding = utils.load_word_embedding()
    num_gpu = torch.cuda.device_count()
    model = Classifier(embedding, args.lstm_hidden_dim,
                       args.num_class, args.n_layers, args.bidirectional, args.dropout_keep_prob)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for GPU usage
    if num_gpu > 1:
        model = nn.DataParallel(model)
        model.to(device)
        loss_ft.cuda()

    val_iter = utils.data_iter(args.seq_len, args.batch_size, args.vocab_size, False, 'valid')
    for epoch in range(args.num_epochs):
        train_iter = utils.data_iter(args.seq_len, args.batch_size, True, 'train')
        start = time.time()
        train_loss, train_acc = train_model(model, train_iter, epoch)
        logger.info("Training elapsed time: %.4fs per epoch\n" % (time.time() - start))

        val_loss, val_acc = eval_model(model, val_iter, epoch, True)
        logger.info('--- Epoch: %d ---' %(epoch+1))
        logger.info('Train Acc: %.2f, Train Loss: %.4f' %(100*train_acc, train_loss))
        logger.info('Val Acc: %.2f, Val Loss: %.4f' %(100*val_acc, val_loss))

def inference(print_datetime):
    best_model = Classifier(embedding, args.lstm_hidden_dim,
                   args.num_class, args.n_layers, args.bidirectional, args.dropout_keep_prob)
    ckpts = glob.glob('./ckpt/%s/*' %print_datetime)
    path = ckpts[np.argsort([float(ckpt.replace('.pt', '').split('_')[-1]) for ckpt in ckpts])[-1]]
    print('Best Model: %s' %path)
    best_model.load_state_dict(torch.load(path))

    test_iter = utils.data_iter(args.seq_len, args.batch_size, args.vocab_size, False, 'test')
    test_loss, test_acc = eval_model(best_model, test_iter)
    print('Test Acc: %.2f, Test Loss: %.4f' % (100 * test_acc, test_loss))

if __name__ == '__main__':
    model_name = 'biLSTM'
    print_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if not os.path.exists('./logs'): os.makedirs('./logs')
    logger = utils.logger_fn("logger", "./logs/train_{0}-{1}.log".format(model_name, print_datetime))
    logger.info('Var Info: seq_len = %d,  hidden_dim: %d\n' %(args.seq_len, args.lstm_hidden_dim))

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('./logs/train_{0}-{1}'.format(model_name, print_datetime))
    # training start
    loss_ft = nn.CrossEntropyLoss()
    main()
    inference(print_datetime)