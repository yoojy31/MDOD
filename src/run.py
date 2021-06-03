import os
import time
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from lib import util as lib_util
import option
import util


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # torch.multiprocessing.set_sharing_strategy('file_system')

    print('[RUN] parse arguments')
    args = option.parse_options()

    print('[RUN] make result directories')
    lib_util.make_dir(args.result_dir_dict['root'])
    util.copy_file(args.bash_file, args.result_dir)
    util.copy_dir('./src', args.result_dir_dict['src'])
    for name in args.result_dir_dict.keys():
        lib_util.make_dir(args.result_dir_dict[name])

    world_size = len(args.run_args['devices'])
    if world_size > 1:
        mp.spawn(ddp_train, args=(world_size, args,), nprocs=world_size, join=True)
    else:
        ddp_train(0, 1, args)


def ddp_train(rank, world_size, args):
    print('rank:', rank, 'world_size:', world_size)
    if world_size > 1:
        setup_ddp(args.run_args, rank, world_size=world_size)

    device_num = args.run_args['devices'][rank]
    with torch.cuda.device(device_num):
        network = option.create_network(args.global_args, args.network_info)
        network = nn.SyncBatchNorm.convert_sync_batchnorm(network) if args.run_args['sync_bnorm'] else network
        network_ddp = nn.parallel.DistributedDataParallel(
            network, device_ids=[device_num], find_unused_parameters=True) \
            if world_size > 1 else network

        post_proc = option.create_post_proc(args.global_args, args.post_proc_info)
        framework = option.create_framework(
            args.global_args, args.framework_info, network, post_proc, 1)
        framework_ddp = option.create_framework(
            args.global_args, args.framework_info, network_ddp, post_proc, world_size)

        optimizer = option.create_optimizer(args.optimizer_info, network)
        load_snapshot(args.run_args['load_dir'], network, optimizer)
        print('[OPTIMIZER] learning rate:', optimizer.param_groups[0]['lr'])
        amp_scaler = amp.GradScaler() if args.run_args['amp'] else None

        train_dataset = option.create_dataset(args.global_args, args.train_data_loader_info)
        test_dataset = option.create_dataset(args.global_args, args.test_data_loader_info)
        train_data_loader = option.create_data_loader(
            train_dataset, args.train_data_loader_info, rank=rank, world_size=world_size)
        test_data_loader = option.create_data_loader(test_dataset, args.test_data_loader_info)

        train_logger = SummaryWriter(args.result_dir_dict['log']) \
            if ((rank == 0) and args.run_args['write_log']) else None

        tester_dict = dict() if rank == 0 else None
        if tester_dict is not None:
            for tester_info in args.tester_info_list:
                tester_dict[tester_info['tester']] = option.create_tester(args.global_args, tester_info)

        n_batches = train_data_loader.__len__()
        global_step = args.run_args['init_epoch'] * n_batches

        for e in range(args.run_args['init_epoch'], args.run_args['max_epoch'] + 1):
            if rank == 0:
                if e in args.run_args['save_epoch_list']:
                    snapshot_dir = os.path.join(args.result_dir_dict['snapshot'], '%03d' % e)
                    save_snapshot(framework.network, optimizer, snapshot_dir)
                else:
                    snapshot_dir = os.path.join(args.result_dir_dict['snapshot'], 'cur')
                    save_snapshot(framework.network, optimizer, snapshot_dir)

            if rank == 0 and (e in args.run_args['test_epoch_list']):
                test_dir = os.path.join(args.result_dir_dict['test'], '%03d' % e)
                run_testers(tester_dict, framework, test_data_loader, test_dir)

            if e == args.run_args['max_epoch']:
                break

            if e in args.run_args['lr_decay_schd'].keys():
                decay_learning_rate(optimizer, args.run_args['lr_decay_schd'][e])

            if e in args.run_args['loss_args_schd'].keys():
                update_loss_args(network.loss_func, args.run_args['loss_args_schd'][e])

            if args.run_args['manually_shuffle']:
                train_data_loader.dataset.shuffle(seed=e)

            start_time = time.time()
            for b, train_data_dict in enumerate(train_data_loader):
                batch_time = time.time() - start_time

                # print('[%d: batch img show]' % b)
                # imgs = train_data_dict['img']
                # for img in imgs:
                #     img = np.transpose(lib_util.cvt_torch2numpy(img), axes=(1, 2, 0))
                #     img = np.clip(img * 0.225 + 0.450, a_min=0.0, a_max=1.0) * 255.0
                #     cv2.imshow("img", img.astype(np.uint8)[:, :, ::-1])
                #     cv2.waitKey(0)
                # continue

                update_flag = True if (global_step + 1) % args.run_args['grad_accum'] == 0 else False
                train_loss_dict, value_dict, train_time = \
                    train_network_one_step(args, framework_ddp, optimizer, train_data_dict,
                                           update=update_flag, amp_scaler=amp_scaler)

                if rank == 0 and (b % args.run_args['print_intv'] == 0):
                    print_str = '[TRAINING] '
                    print_str += '%d/%d, %d/%d, %d | ' % (e, args.run_args['max_epoch'], b, n_batches, global_step)
                    print_str += 'b-time: %0.3f, t-time: %0.3f | ' % (batch_time, train_time)
                    print_str += util.cvt_dict2str(train_loss_dict)

                    if train_logger is not None:
                        for key, value in train_loss_dict.items():
                            train_logger.add_scalar(key, value, global_step)

                    if len(value_dict.keys()) > 0:
                        print_str += (' | ' + util.cvt_dict2str(value_dict))

                        if train_logger is not None:
                            for key, value in value_dict.items():
                                train_logger.add_scalar(key, value, global_step)
                    print(print_str)

                # train_loss_dict.clear()
                # train_data_dict.clear()
                # del train_loss_dict, train_data_dict

                global_step += 1
                start_time = time.time()

        if world_size > 1:
            cleanup_ddp()


def setup_ddp(run_args, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(run_args['port'])
    # os.environ['WORLD_SIZE'] = world_size
    # os.environ['RANK'] = rank
    # initialize the process group
    dist.init_process_group('nccl', init_method='env://', rank=rank, world_size=world_size)
    # dist.init_process_group('gloo', rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def load_snapshot(load_dir, network, optimizer):
    if (load_dir is not None) and os.path.exists(load_dir):
        network_path = os.path.join(load_dir, 'network.pth')
        network.load(network_path)

        optimizer_path = os.path.join(load_dir, 'optimizer.pth')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
            print('[OPTIMIZER] load: %s' % optimizer_path)


def train_network_one_step(args, framework, optimizer, train_data_dict, update=True, amp_scaler=None):
    max_grad = args.run_args['max_grad']

    start_time = time.time()
    if args.run_args['amp']:
        with amp.autocast():
            _, train_loss_dict, value_dict = framework.train_forward(train_data_dict)
    else:
        # print('amp false')
        _, train_loss_dict, value_dict = framework.train_forward(train_data_dict)

    if update:
        update_network(framework.network, optimizer, train_loss_dict, max_grad, amp_scaler)
    else:
        backward_network(optimizer, train_loss_dict)
    train_time = time.time() - start_time

    return train_loss_dict, value_dict, train_time


def update_network(network, optimizer, loss_dict, max_grad=None, amp_scaler=None):
    sum_loss = 0
    for _, loss in loss_dict.items():
        if torch.isnan(loss):
            pass
        else:
            sum_loss += loss

    if amp_scaler is None:
        # print('amp_scalar is None')
        if sum_loss != 0:
            sum_loss.backward()
        if max_grad is not None:
            nn.utils.clip_grad_norm_(network.parameters(), max_grad)

        optimizer.step()
        optimizer.zero_grad()
    else:
        if sum_loss != 0:
            amp_scaler.scale(sum_loss).backward()
        if max_grad is not None:
            nn.utils.clip_grad_norm_(network.parameters(), max_grad)

        amp_scaler.step(optimizer)
        # Updates the scale for next iteration
        amp_scaler.update()
        optimizer.zero_grad()



def backward_network(optimizer, loss_dict):
    sum_loss = 0
    for _, loss in loss_dict.items():
        sum_loss += loss
    # optimizer.zero_grad()
    if sum_loss != 0:
        sum_loss.backward()


def save_snapshot(network, optimizer, save_dir):
    lib_util.make_dir(save_dir)
    network_path = os.path.join(save_dir, 'network.pth')
    optimizer_path = os.path.join(save_dir, 'optimizer.pth')
    network.save(network_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    print('[OPTIMIZER] save: %s\n' % optimizer_path)


def run_testers(tester_dict, framework, test_data_loader, test_dir):
    lib_util.make_dir(test_dir)
    for key, tester in tester_dict.items():
        # test_data_loader = create_data_loader(test_dataset, args.test_data_loader_info)
        tester_dir = os.path.join(test_dir, key)
        tester.run(framework, test_data_loader, tester_dir)
        # test_data_loader.stop()
        # del test_data_loader
        print('[TEST] %s: %s' % (key, tester_dir))
    print('')


def decay_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        new_lr = old_lr * decay_rate
        param_group['lr'] = new_lr
    print('[OPTIMIZER] learning rate: %f -> %f' % (old_lr, new_lr))


def update_loss_args(loss_func, new_loss_args):
    # old_lw_dict_str = util.cvt_dict2str(loss_func.lw_dict)
    loss_func.update(new_loss_args)
    # new_lw_dict_str = util.cvt_dict2str(loss_func.lw_dict)
    # print('[LOSS FUNCTION] loss weight:\n{%s} -> {%s}\n' % (old_lw_dict_str, new_lw_dict_str))
    # new_loss_args_str = util.cvt_dict2str(new_loss_args)
    # print('[LOSS FUNCTION] new loss args: %s\n' % new_loss_args_str)


def create_data_loader(dataset, data_loader_info):
    batch_size = data_loader_info['batch_size']
    shuffle = data_loader_info['shuffle']
    num_workers = data_loader_info['num_workers']
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    data_loader.start()
    return data_loader


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()
