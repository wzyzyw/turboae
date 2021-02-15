__author__ = 'yihanjiang'
# update 10/18/2019, code to replicate TurboAE paper in NeurIPS 2019.
# Tested on PyTorch 1.0.
# TBD: remove all non-TurboAE related functions.
# 网络模型只包括译码器，编码器不涉及网络，原代码中的train已经用system重构了，encoders中也重写了编码函数，但main中模型参数读取等还未修改。
import torch
import torch.optim as optim
import numpy as np
import sys
from get_args import get_args
from trainer import train, validate, test

from numpy import arange
from numpy.random import mtrand
import torch
import torch.nn.functional as F
import commpy.channelcoding.interleavers as RandInterlv
import numpy as np

from ste import STEQuantize as MyQuantize
from utils import snr_sigma2db, snr_db2sigma, code_power, errors_ber_pos, errors_ber, errors_bler
from loss import customized_loss
from channels import generate_noise

import numpy as np
from numpy import arange
from numpy.random import mtrand
# utils for logger
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def import_enc(args):
    # choose encoder

    if args.encoder == 'TurboAE_rate3_rnn':
        from encoders import ENC_interRNN as ENC

    elif args.encoder in ['TurboAE_rate3_cnn', 'TurboAE_rate3_cnn_dense']:
        from encoders import ENC_interCNN as ENC

    elif args.encoder == 'turboae_2int':
        from encoders import ENC_interCNN2Int as ENC

    elif args.encoder == 'rate3_cnn':
        from encoders import CNN_encoder_rate3 as ENC

    elif args.encoder in ['TurboAE_rate3_cnn2d', 'TurboAE_rate3_cnn2d_dense']:
        from encoders import ENC_interCNN2D as ENC

    elif args.encoder == 'TurboAE_rate3_rnn_sys':
        from encoders import ENC_interRNN_sys as ENC

    elif args.encoder == 'TurboAE_rate2_rnn':
        from encoders import ENC_turbofy_rate2 as ENC

    elif args.encoder == 'TurboAE_rate2_cnn':
        from encoders import ENC_turbofy_rate2_CNN as ENC  # not done yet

    elif args.encoder in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
        from encoders import ENC_TurboCode2 as ENC          # DeepTurbo, encoder not trainable.

    elif args.encoder == 'rate3_cnn2d':
        from encoders import ENC_CNN2D as ENC

    else:
        print('Unknown Encoder, stop')

    return ENC

def import_dec(args):

    if args.decoder == 'TurboAE_rate2_rnn':
        from decoders import DEC_LargeRNN_rate2 as DEC

    elif args.decoder == 'TurboAE_rate2_cnn':
        from decoders import DEC_LargeCNN_rate2 as DEC  # not done yet

    elif args.decoder in ['TurboAE_rate3_cnn', 'TurboAE_rate3_cnn_dense']:
        from decoders import DEC_LargeCNN as DEC

    elif args.decoder == 'turboae_2int':
        from decoders import DEC_LargeCNN2Int as DEC

    elif args.encoder == 'rate3_cnn':
        from decoders import CNN_decoder_rate3 as DEC

    elif args.decoder in ['TurboAE_rate3_cnn2d', 'TurboAE_rate3_cnn2d_dense']:
        from decoders import DEC_LargeCNN2D as DEC

    elif args.decoder == 'TurboAE_rate3_rnn':
        from decoders import DEC_LargeRNN as DEC

    elif args.decoder == 'nbcjr_rate3':                # ICLR 2018 paper
        from decoders import NeuralTurbofyDec as DEC

    elif args.decoder == 'rate3_cnn2d':
        from decoders import DEC_CNN2D as DEC

    return DEC
def system(args, optimizer,enc, dec,use_cuda = False, verbose = True):
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loss = 0.0
    for batch_idx in range(int(args.num_block/args.batch_size)):
        if args.is_variable_block_len:
            block_len = np.random.randint(args.block_len_low, args.block_len_high)
        else:
            block_len = args.block_len
        optimizer.zero_grad()
        # generate bit and noise
        X_train    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)
        fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')
        X_train, fwd_noise = X_train.to(device), fwd_noise.to(device)
        # pass system
        if args.is_interleave == 0:
            pass
        elif args.is_same_interleaver == 0:
            interleaver = RandInterlv.RandInterlv(args.block_len, np.random.randint(0, 1000))

            p_array = interleaver.p_array
            enc.set_interleaver(p_array)
            dec.set_interleaver(p_array)
        else:# self.args.is_same_interleaver == 1
            interleaver = RandInterlv.RandInterlv(args.block_len, 0) # not random anymore!
            p_array = interleaver.p_array
            enc.set_interleaver(p_array)
            dec.set_interleaver(p_array)
        codes  = enc.encode(X_train)
        if self.args.channel in ['awgn', 't-dist', 'radar', 'ge_awgn','bikappa']:
            # print("noise_type:",self.args.channel)
            received_codes = codes + fwd_noise

        elif self.args.channel == 'bec':
            received_codes = codes * fwd_noise

        elif self.args.channel in ['bsc', 'ge']:
            received_codes = codes * (2.0*fwd_noise - 1.0)
            received_codes = received_codes.type(torch.FloatTensor)
        else:
            print('default AWGN channel')
            received_codes = codes + fwd_noise
        if args.rec_quantize:
            myquantize = MyQuantize.apply
            received_codes = myquantize(received_codes, args.rec_quantize_level, args.rec_quantize_level)
        x_dec = dec(received_codes)
        loss = customized_loss(output, X_train, args, noise=fwd_noise, code = code)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss = train_loss /(args.num_block/args.batch_size)
    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss), \
            ' running time', str(end_time - start_time))


if __name__ == '__main__':
    #################################################
    # load args & setup logger
    #################################################
    identity = str(np.random.random())[2:8]
    print('[ID]', identity)

    # put all printed things to log file
    logfile = open('./logs/'+identity+'_log.txt', 'a')
    sys.stdout = Logger('./logs/'+identity+'_log.txt', sys.stdout)

    args = get_args()
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #################################################
    # Setup Channel AE: Encoder, Decoder, Channel
    #################################################
    # choose encoder and decoder.
    ENC = import_enc(args)
    DEC = import_dec(args)

    # setup interleaver.
    # 代码中两次设置了p_array,第一次是这里，目的是初始化enc和dec内的interleaver对象（传统编译码器是没有用的），在channelae中会重新设置一次p_array，所以最后是以第二次设置为准，这次和nerualbcjr是相同的，因此，在scfde中没必要设置两次，直接一次完成，后续的话可能需要在scfde中补充根据不同情形下的交织器
    if args.is_interleave == 1:           # fixed interleaver.
        seed = np.random.randint(0, 1)
        rand_gen = mtrand.RandomState(seed)
        p_array1 = rand_gen.permutation(arange(args.block_len))
        p_array2 = rand_gen.permutation(arange(args.block_len))

    elif args.is_interleave == 0:
        p_array1 = range(args.block_len)   # no interleaver.
        p_array2 = range(args.block_len)   # no interleaver.
    else:
        seed = np.random.randint(0, args.is_interleave)
        rand_gen = mtrand.RandomState(seed)
        p_array1 = rand_gen.permutation(arange(args.block_len))
        seed = np.random.randint(0, args.is_interleave)
        rand_gen = mtrand.RandomState(seed)
        p_array2 = rand_gen.permutation(arange(args.block_len))

    print('using random interleaver', p_array1, p_array2)
    # 要同时使用p_array1, p_array2必须编码器和译码器都是turboae_2int，在scfde中，不考虑编码器，所以这里if的内容可以删除
    if args.encoder == 'turboae_2int' and args.decoder == 'turboae_2int':
        encoder = ENC(args, p_array1, p_array2)
        decoder = DEC(args, p_array1, p_array2)
    else:
        encoder = ENC(args, p_array1)
        decoder = DEC(args, p_array1)

    # choose support channels

    from channel_ae import Channel_AE
    model = Channel_AE(args, encoder, decoder).to(device)
    # from channel_ae import Channel_ModAE
    # model = Channel_ModAE(args, encoder, decoder).to(device)


    # make the model parallel
    if args.is_parallel == 1:
        model.enc.set_parallel()
        model.dec.set_parallel()

    # weight loading
    if args.init_nw_weight == 'default':
        pass

    else:
        pretrained_model = torch.load(args.init_nw_weight,map_location=torch.device('cpu'))

        try:
            model.load_state_dict(pretrained_model.state_dict(), strict = False)

        except:
            model.load_state_dict(pretrained_model, strict = False)

        model.args = args

    print(model)


    ##################################################################
    # Setup Optimizers, only Adam and Lookahead for now.
    ##################################################################

    if args.optimizer == 'lookahead':
        print('Using Lookahead Optimizers')
        from optimizers import Lookahead
        lookahead_k = 5
        lookahead_alpha = 0.5
        if args.num_train_enc != 0 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']: # no optimizer for encoder
            enc_base_opt  = optim.Adam(model.enc.parameters(), lr=args.enc_lr)
            enc_optimizer = Lookahead(enc_base_opt, k=lookahead_k, alpha=lookahead_alpha)

        if args.num_train_dec != 0:
            dec_base_opt  = optim.Adam(filter(lambda p: p.requires_grad, model.dec.parameters()), lr=args.dec_lr)
            dec_optimizer = Lookahead(dec_base_opt, k=lookahead_k, alpha=lookahead_alpha)

        general_base_opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.dec_lr)
        general_optimizer = Lookahead(general_base_opt, k=lookahead_k, alpha=lookahead_alpha)

    else: # Adam, SGD, etc....
        if args.optimizer == 'adam':
            OPT = optim.Adam
        elif args.optimizer == 'sgd':
            OPT = optim.SGD
        else:
            OPT = optim.Adam

        if args.num_train_enc != 0 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']: # no optimizer for encoder
            enc_optimizer = OPT(model.enc.parameters(),lr=args.enc_lr)

        if args.num_train_dec != 0:
            dec_optimizer = OPT(filter(lambda p: p.requires_grad, model.dec.parameters()), lr=args.dec_lr)

        general_optimizer = OPT(filter(lambda p: p.requires_grad, model.parameters()),lr=args.dec_lr)

    #################################################
    # Training Processes
    #################################################
    report_loss, report_ber = [], []

    for epoch in range(1, args.num_epoch + 1):

        if args.joint_train == 1 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
            for idx in range(args.num_train_enc+args.num_train_dec):
                train(epoch, model, general_optimizer, args, use_cuda = use_cuda, mode ='encoder')

        else:
            if args.num_train_enc > 0 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
                for idx in range(args.num_train_enc):
                    train(epoch, model, enc_optimizer, args, use_cuda = use_cuda, mode ='encoder')

            if args.num_train_dec > 0:
                for idx in range(args.num_train_dec):
                    train(epoch, model, dec_optimizer, args, use_cuda = use_cuda, mode ='decoder')

        this_loss, this_ber  = validate(model, general_optimizer, args, use_cuda = use_cuda)
        report_loss.append(this_loss)
        report_ber.append(this_ber)
    if args.print_test_traj == True:
        print('test loss trajectory', report_loss)
        print('test ber trajectory', report_ber)
        print('total epoch', args.num_epoch)

    #################################################
    # Testing Processes
    #################################################

    torch.save(model.state_dict(), './tmp/torch_model_'+identity+'.pt')
    print('saved model', './tmp/torch_model_'+identity+'.pt')


    if args.is_variable_block_len:
        # 测试时，选择是否改变编码块的长度
        print('testing block length',args.block_len_low )
        test(model, args, block_len=args.block_len_low, use_cuda = use_cuda)
        print('testing block length',args.block_len )
        test(model, args, block_len=args.block_len, use_cuda = use_cuda)
        print('testing block length',args.block_len_high )
        test(model, args, block_len=args.block_len_high, use_cuda = use_cuda)

    else:
        test(model, args, use_cuda = use_cuda)














