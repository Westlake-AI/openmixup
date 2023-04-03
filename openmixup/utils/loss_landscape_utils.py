# Visualizing the Loss Landscape of Neural Nets (NeurIPS, 2018)
# Modified from https://github.com/tomgoldstein/loss-landscape

import copy
import os
from os.path import exists, commonprefix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
import torch
try:
    import h5py
except ImportError:
    h5py = None  # pip install h5py

from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.parallel import is_module_wrapper
from sklearn.decomposition import PCA


""" https://github.com/tomgoldstein/loss-landscape/h5_util.py

    Serialization and deserialization of directions in the direction file.
"""

def write_list(f, name, direction):
    """ Save the direction to the hdf5 file with name as the key

        Args:
            f: h5py file object
            name: key name_surface_file
            direction: a list of tensors
    """

    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, torch.Tensor):
            l = l.cpu().numpy()
        grp.create_dataset(str(i), data=l)


def read_list(f, name):
    """ Read group with name as the key from the hdf5 file and return a list numpy vectors. """
    grp = f[name]
    return [grp[str(i)] for i in range(len(grp))]



""" https://github.com/tomgoldstein/loss-landscape/net_plotter.py

    Manipulate network parameters and setup random directions with normalization.
"""

################################################################################
#                 Supporting functions for weights manipulation
################################################################################
def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


def set_weights(net, weights, directions=None, step=None):
    """ Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type_as(p.data))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        for (p, w, d) in zip(net.parameters(), weights, changes):
            p.data = w + torch.Tensor(d).type_as(w)


def set_states(net, states, directions=None, step=None):
    """ Overwrite the network's state_dict or change it along directions with a step size. """
    if directions is None:
        load_state_dict(net, states)
    else:
        assert step is not None, 'If direction is provided then the step must be specified as well'
        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        new_states = copy.deepcopy(states)
        assert (len(new_states) == len(changes))
        for (k, v), d in zip(new_states.items(), changes):
            d = torch.tensor(d).type_as(v)
            v.add_(d)

        load_state_dict(net, new_states)


def get_random_weights(weights):
    """ Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size(), device=w.device) for w in weights]


def get_random_states(states):
    """ Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    return [torch.randn(w.size(), device=w.device) for k, w in states.items()]


def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]


def get_diff_states(states, states2):
    """ Produce a direction from 'states' to 'states2'."""
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]


################################################################################
#                        Normalization Functions
################################################################################
def normalize_direction(direction, weights, norm='filter'):
    """ Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """ The normalization scales the direction entries according to the entries of weights. """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def normalize_directions_for_states(direction, states, norm='filter', ignore='ignore'):
    assert(len(direction) == len(states))
    for d, (k, w) in zip(direction, states.items()):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def ignore_biasbn(directions):
    """ Set bias and bn parameters in directions to zero """
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)

################################################################################
#                       Create directions
################################################################################
def create_target_direction(net, net2, dir_type='states'):
    """ Setup a target direction from one model to the other

        Args:
          net: the source model
          net2: the target model with the same architecture as net.
          dir_type: 'weights' or 'states', type of directions.

        Returns:
          direction: the target direction from net to net2 with the same dimension
                     as weights or states.
    """

    assert (net2 is not None)
    # direction between net2 and net
    if dir_type == 'weights':
        w = get_weights(net)
        w2 = get_weights(net2)
        direction = get_diff_weights(w, w2)
    elif dir_type == 'states':
        s = net.state_dict()
        s2 = net2.state_dict()
        direction = get_diff_states(s, s2)

    return direction


def create_random_direction(net, dir_type='weights', ignore='biasbn', norm='filter'):
    """ Setup a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
    if dir_type == 'weights':
        weights = get_weights(net) # a list of parameters.
        direction = get_random_weights(weights)
        normalize_directions_for_weights(direction, weights, norm, ignore)
    elif dir_type == 'states':
        states = net.state_dict() # a dict of parameters, including BN's running mean/var.
        direction = get_random_states(states)
        normalize_directions_for_states(direction, states, norm, ignore)

    return direction


def setup_direction(args, dir_file, net):
    """ Setup the h5 file to store the directions.
        - xdirection, ydirection: The pertubation direction added to the mdoel.
          The direction is a list of tensors.
    """
    print('-------------------------------------------------------------------')
    print('setup_direction')
    print('-------------------------------------------------------------------')
    
    # Setup env for preventing lock on h5py file for newer h5py versions
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    # Skip if the direction file already exists
    if exists(dir_file):
        f = h5py.File(dir_file, 'r')
        if (args.y and 'ydirection' in f.keys()) or 'xdirection' in f.keys():
            f.close()
            print ("%s is already setted up" % dir_file)
            return
        f.close()

    # Create the plotting directions
    f = h5py.File(dir_file,'w') # create file, fail if exists
    if not args.dir_file:
        print("Setting up the plotting directions...")
        if args.model_file2:
            net2 = copy.deepcopy(net)
            if is_module_wrapper(net2):
                load_checkpoint(net2.module, args.model_file2, map_location='cpu')
            else:
                load_checkpoint(net2, args.model_file2, map_location='cpu')
            net2 = net2.cuda()
            xdirection = create_target_direction(net, net2, args.dir_type)
        else:
            xdirection = create_random_direction(net, args.dir_type, args.xignore, args.xnorm)
        write_list(f, 'xdirection', xdirection)

        if args.y:
            if args.same_dir:
                ydirection = xdirection
            elif args.model_file3:
                net3 = copy.deepcopy(net)
                if is_module_wrapper(net3):
                    load_checkpoint(net3.module, args.model_file3, map_location='cpu')
                else:
                    load_checkpoint(net3, args.model_file3, map_location='cpu')
                net3 = net3.cuda()
                ydirection = create_target_direction(net, net3, args.dir_type)
            else:
                ydirection = create_random_direction(net, args.dir_type, args.yignore, args.ynorm)
            write_list(f, 'ydirection', ydirection)

    f.close()
    print ("direction file created: %s" % dir_file)


def name_direction_file(args):
    """ Name the direction file that stores the random directions. """

    if args.dir_file:
        assert exists(args.dir_file), "%s does not exist!" % args.dir_file
        return args.dir_file

    dir_file = ""  # default

    file1, file2, file3 = args.checkpoint, args.model_file2, args.model_file3

    # name for xdirection
    if file2:
        # 1D linear interpolation between two models
        assert exists(file2), file2 + " does not exist!"
        if file1[:file1.rfind('/')] == file2[:file2.rfind('/')]:
            # model_file and model_file2 are under the same folder
            dir_file += file1 + '_' + file2[file2.rfind('/')+1:]
        else:
            # model_file and model_file2 are under different folders
            prefix = commonprefix([file1, file2])
            prefix = prefix[0:prefix.rfind('/')]
            dir_file += file1[:file1.rfind('/')] + '_' + file1[file1.rfind('/')+1:] + '_' + \
                       file2[len(prefix)+1: file2.rfind('/')] + '_' + file2[file2.rfind('/')+1:]
    else:
        dir_file += file1

    dir_file += '_' + args.dir_type
    if args.xignore:
        dir_file += '_xignore=' + args.xignore
    if args.xnorm:
        dir_file += '_xnorm=' + args.xnorm

    # name for ydirection
    if args.y:
        if file3:
            assert exists(file3), "%s does not exist!" % file3
            if file1[:file1.rfind('/')] == file3[:file3.rfind('/')]:
               dir_file += file3
            else:
               # model_file and model_file3 are under different folders
               dir_file += file3[:file3.rfind('/')] + '_' + file3[file3.rfind('/')+1:]
        else:
            if args.yignore:
                dir_file += '_yignore=' + args.yignore
            if args.ynorm:
                dir_file += '_ynorm=' + args.ynorm
            if args.same_dir: # ydirection is the same as xdirection
                dir_file += '_same_dir'

    # index number
    if args.idx > 0: dir_file += '_idx=' + str(args.idx)

    dir_file += ".h5"

    return dir_file


def load_directions(dir_file):
    """ Load direction(s) from the direction file."""

    f = h5py.File(dir_file, 'r')
    if 'ydirection' in f.keys():  # If this is a 2D plot
        xdirection = read_list(f, 'xdirection')
        ydirection = read_list(f, 'ydirection')
        directions = [xdirection, ydirection]
    else:
        directions = [read_list(f, 'xdirection')]

    return directions



""" https://github.com/tomgoldstein/loss-landscape/projection.py
    Project a model or multiple models to a plane spaned by given directions.
"""

def tensorlist_to_tensor(weights):
    """ Concatnate a list of tensors into one tensor (on CPU).

        Args:
            weights: a list of parameter tensors, e.g. net_plotter.get_weights(net).

        Returns:
            concatnated 1D tensor
    """
    return torch.cat(
        [w.view(w.numel()).cpu() if w.dim() > 1 else torch.FloatTensor(w.cpu()) for w in weights])


def nplist_to_tensor(nplist):
    """ Concatenate a list of numpy vectors into one tensor.

        Args:
            nplist: a list of numpy vectors, e.g., direction loaded from h5 file.

        Returns:
            concatnated 1D tensor
    """
    v = []
    for d in nplist:
        w = torch.tensor(d*np.float64(1.0))
        # Ignoreing the scalar values (w.dim() = 0).
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def npvec_to_tensorlist(direction, params):
    """ Convert a numpy vector to a list of tensors with the same shape as "params".

        Args:
            direction: a list of numpy vectors, e.g., a direction loaded from h5 file.
            base: a list of parameter tensors from net

        Returns:
            a list of tensors with the same shape as base
    """
    if isinstance(params, list):
        w2 = copy.deepcopy(params)
        idx = 0
        for w in w2:
            w.copy_(torch.tensor(direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert(idx == len(direction))
        return w2
    else:
        s2 = []
        idx = 0
        for (k, w) in params.items():
            s2.append(torch.Tensor(direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert(idx == len(direction))
        return s2


def cal_angle(vec1, vec2):
    """ Calculate cosine similarities between two torch tensors or two ndarraies
        Args:
            vec1, vec2: two tensors or numpy ndarraies
    """
    if isinstance(vec1, torch.Tensor) and isinstance(vec1, torch.Tensor):
        return torch.dot(vec1, vec2)/(vec1.norm()*vec2.norm()).item()
    elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
        return np.ndarray.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def project_1D(w, d):
    """ Project vector w to vector d and get the length of the projection.

        Args:
            w: vectorized weights
            d: vectorized direction

        Returns:
            the projection scalar
    """
    assert len(w) == len(d), 'dimension does not match for w and '
    scale = torch.dot(w, d)/d.norm()
    return scale.item()


def project_2D(d, dx, dy, proj_method):
    """ Project vector d to the plane spanned by dx and dy.

        Args:
            d: vectorized weights
            dx: vectorized direction
            dy: vectorized direction
            proj_method: projection method
        Returns:
            x, y: the projection coordinates
    """

    if proj_method == 'cos':
        # when dx and dy are orthorgonal
        x = project_1D(d, dx)
        y = project_1D(d, dy)
    elif proj_method == 'lstsq':
        # solve the least squre problem: Ax = d
        A = np.vstack([dx.numpy(), dy.numpy()]).T
        [x, y] = np.linalg.lstsq(A, d.numpy())[0]

    return x, y


def project_trajectory(dir_file, w, s, model, model_files,
                       dir_type='weights', proj_method='cos'):
    """ Project the optimization trajectory onto the given two directions.

        Args:
          dir_file: the h5 file that contains the directions
          w: weights of the final model
          s: states of the final model
          model: the network model (nn.Module)
          model_files: list of the checkpoint files
          dir_type: the type of the direction, weights or states
          proj_method: cosine projection

        Returns:
          proj_file: the projection filename
    """

    proj_file = dir_file + '_proj_' + proj_method + '.npy'
    if os.path.exists(proj_file):
        print('The projection file exists! No projection is performed unless %s is deleted' % proj_file)
        return proj_file

    # read directions and convert them to vectors
    directions = load_directions(dir_file)
    dx = nplist_to_tensor(directions[0])
    dy = nplist_to_tensor(directions[1])

    xcoord, ycoord = [], []
    for model_file in model_files:
        net2 = copy.deepcopy(model)
        if is_module_wrapper(net2):
            load_checkpoint(net2.module, model_file, map_location='cpu')
        else:
            load_checkpoint(net2, model_file, map_location='cpu')
        net2 = net2.cuda()
        if dir_type == 'weights':
            w2 = get_weights(net2)
            d = get_diff_weights(w, w2)
        elif dir_type == 'states':
            s2 = net2.state_dict()
            d = get_diff_states(s, s2)
        d = tensorlist_to_tensor(d)

        x, y = project_2D(d, dx, dy, proj_method)
        print ("%s  (%.4f, %.4f)" % (model_file, x, y))

        xcoord.append(x)
        ycoord.append(y)

    proj_dict = dict(proj_xcoord=np.array(xcoord),
                     proj_ycoord=np.array(ycoord))
    np.save(proj_file, proj_dict)

    return proj_file


def setup_PCA_directions(args, model, model_files, w, s):
    """
        Find PCA directions for the optimization path from the initial model
        to the final trained model.

        Returns:
            dir_name: the h5 file that stores the directions.
    """

    # Name the .h5 file that stores the PCA directions.
    folder_name = args.model_folder + '/PCA_' + args.dir_type
    if args.xignore or args.yignore:
        args.ignore = args.xignore if args.xignore else args.yignore
        folder_name += '_ignore=' + args.ignore
    folder_name += '_save_interval=' + str(args.save_interval)
    os.system('mkdir ' + folder_name)
    dir_name = folder_name + '/directions.h5'

    # skip if the direction file exists
    if os.path.exists(dir_name):
        f = h5py.File(dir_name, 'a')
        if 'explained_variance_' in f.keys():
            f.close()
            return dir_name

    # load models and prepare the optimization path matrix
    matrix = []
    for model_file in model_files:
        print (model_file)
        net2 = copy.deepcopy(model)
        if is_module_wrapper(net2):
            load_checkpoint(net2.module, model_file, map_location='cpu')
        else:
            load_checkpoint(net2, model_file, map_location='cpu')
        net2 = net2.cuda()
        if args.dir_type == 'weights':
            w2 = get_weights(net2)
            d = get_diff_weights(w, w2)
        elif args.dir_type == 'states':
            s2 = net2.state_dict()
            d = get_diff_states(s, s2)
        if args.ignore == 'biasbn':
            ignore_biasbn(d)
        d = tensorlist_to_tensor(d)
        matrix.append(d.cpu().numpy())

    # Perform PCA on the optimization path matrix
    print ("Perform PCA on the models")
    pca = PCA(n_components=2)
    pca.fit(np.array(matrix))
    pc1 = np.array(pca.components_[0])
    pc2 = np.array(pca.components_[1])
    print("angle between pc1 and pc2: %f" % cal_angle(pc1, pc2))

    print("pca.explained_variance_ratio_: %s" % str(pca.explained_variance_ratio_))

    # convert vectorized directions to the same shape as models to save in h5 file.
    if args.dir_type == 'weights':
        xdirection = npvec_to_tensorlist(pc1, w)
        ydirection = npvec_to_tensorlist(pc2, w)
    elif args.dir_type == 'states':
        xdirection = npvec_to_tensorlist(pc1, s)
        ydirection = npvec_to_tensorlist(pc2, s)

    if args.ignore == 'biasbn':
        ignore_biasbn(xdirection)
        ignore_biasbn(ydirection)

    f = h5py.File(dir_name, 'w')
    write_list(f, 'xdirection', xdirection)
    write_list(f, 'ydirection', ydirection)

    f['explained_variance_ratio_'] = pca.explained_variance_ratio_
    f['singular_values_'] = pca.singular_values_
    f['explained_variance_'] = pca.explained_variance_

    f.close()
    print ('PCA directions saved in: %s' % dir_name)

    return dir_name



""" https://github.com/tomgoldstein/loss-landscape/plot_1D.py

    1D plotting routines
"""

def plot_1d_loss_err(surf_file, xmin=-1.0, xmax=1.0, loss_max=5, log=False,
                     format='pdf', show=False):
    print('------------------------------------------------------------------')
    print('plot_1d_loss_err')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    print(f.keys())
    x = f['xcoordinates'][:]
    assert 'train_loss' in f.keys(), "'train_loss' does not exist"
    train_loss = f['train_loss'][:]
    train_acc = f['train_acc'][:]

    print("train_loss")
    print(train_loss)
    print("train_acc")
    print(train_acc)

    xmin = xmin if xmin != -1.0 else min(x)
    xmax = xmax if xmax != 1.0 else max(x)

    # loss and accuracy map
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if log:
        tr_loss, = ax1.semilogy(x, train_loss, 'b-', label='Training loss', linewidth=1)
    else:
        tr_loss, = ax1.plot(x, train_loss, 'b-', label='Training loss', linewidth=1)
    tr_acc, = ax2.plot(x, train_acc, 'r-', label='Training accuracy', linewidth=1)

    if 'test_loss' in f.keys():
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]
        if log:
            te_loss, = ax1.semilogy(x, test_loss, 'b--', label='Test loss', linewidth=1)
        else:
            te_loss, = ax1.plot(x, test_loss, 'b--', label='Test loss', linewidth=1)
        te_acc, = ax2.plot(x, test_acc, 'r--', label='Test accuracy', linewidth=1)

    plt.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
    ax1.tick_params('y', colors='b', labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    ax1.set_ylim(0, loss_max)
    ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
    ax2.tick_params('y', colors='r', labelsize='x-large')
    ax2.set_ylim(0, 100)
    plt.savefig(surf_file + '_1d_loss_acc' + ('_log' if log else '') + f'.{format}',
                dpi=300, bbox_inches='tight', format=format)

    # train_loss curve
    plt.figure()
    if log:
        plt.semilogy(x, train_loss)
    else:
        plt.plot(x, train_loss)
    plt.ylabel('Training Loss', fontsize='xx-large')
    plt.xlim(xmin, xmax)
    plt.ylim(0, loss_max)
    plt.savefig(surf_file + '_1d_train_loss' + ('_log' if log else '') + f'.{format}',
                dpi=300, bbox_inches='tight', format=format)

    # train_err curve
    plt.figure()
    plt.plot(x, 100 - train_acc)
    plt.xlim(xmin, xmax)
    plt.ylim(0, 100)
    plt.ylabel('Training Error', fontsize='xx-large')
    plt.savefig(surf_file + '_1d_train_err.' + format,
               dpi=300, bbox_inches='tight', format=format)

    if show:
        plt.show()
    f.close()


def plot_1d_loss_err_repeat(prefix, idx_min=1, idx_max=10, xmin=-1.0, xmax=1.0,
                            loss_max=5, format='pdf', show=False):
    """ Plotting multiple 1D loss surface with different directions in one figure. """

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for idx in range(idx_min, idx_max + 1):
        # The file format should be prefix_{idx}.h5
        f = h5py.File(prefix + '_' + str(idx) + '.h5','r')

        x = f['xcoordinates'][:]
        train_loss = f['train_loss'][:]
        train_acc = f['train_acc'][:]
        test_loss = f['test_loss'][:]
        test_acc = f['test_acc'][:]

        xmin = xmin if xmin != -1.0 else min(x)
        xmax = xmax if xmax != 1.0 else max(x)

        tr_loss, = ax1.plot(x, train_loss, 'b-', label='Training loss', linewidth=1)
        te_loss, = ax1.plot(x, test_loss, 'b--', label='Testing loss', linewidth=1)
        tr_acc, = ax2.plot(x, train_acc, 'r-', label='Training accuracy', linewidth=1)
        te_acc, = ax2.plot(x, test_acc, 'r--', label='Testing accuracy', linewidth=1)

    plt.xlim(xmin, xmax)
    ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
    ax1.tick_params('y', colors='b', labelsize='x-large')
    ax1.tick_params('x', labelsize='x-large')
    ax1.set_ylim(0, loss_max)
    ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
    ax2.tick_params('y', colors='r', labelsize='x-large')
    ax2.set_ylim(0, 100)
    plt.savefig(prefix + '_1d_loss_err_repeat.' + format,
               dpi=300, bbox_inches='tight', format=format)

    if show:
        plt.show()


def plot_1d_eig_ratio(surf_file, xmin=-1.0, xmax=1.0, val_1='min_eig', val_2='max_eig', ymax=1,
                      format='pdf', show=False):
    print('------------------------------------------------------------------')
    print('plot_1d_eig_ratio')
    print('------------------------------------------------------------------')

    f = h5py.File(surf_file,'r')
    x = f['xcoordinates'][:]

    Z1 = np.array(f[val_1][:])
    Z2 = np.array(f[val_2][:])
    abs_ratio = np.absolute(np.divide(Z1, Z2))

    plt.plot(x, abs_ratio)
    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)
    plt.savefig(surf_file + '_1d_eig_abs_ratio.' + format,
               dpi=300, bbox_inches='tight', format=format)

    ratio = np.divide(Z1, Z2)
    plt.plot(x, ratio)
    plt.xlim(xmin, xmax)
    plt.ylim(0, ymax)
    plt.savefig(surf_file + '_1d_eig_ratio.' + format,
               dpi=300, bbox_inches='tight', format=format)

    f.close()
    if show:
        plt.show()



""" https://github.com/tomgoldstein/loss-landscape/plot_2D.py

    2D plotting funtions
"""

def plot_2d_contour(surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5,
                    format='pdf', show=False):
    """Plot 2D contour map and 3D surface."""

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
    print(Z)

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    # --------------------------------------------------------------------
    # Plot 2D contours
    # --------------------------------------------------------------------
    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(surf_file + '_' + surf_name + '_2dcontour' + f'.{format}', dpi=300,
                bbox_inches='tight', format=format)

    fig = plt.figure()
    print(surf_file + '_' + surf_name + '_2dcontourf' + f'.{format}')
    CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    fig.savefig(surf_file + '_' + surf_name + '_2dcontourf' + f'.{format}', dpi=300,
                bbox_inches='tight', format=format)

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    fig = plt.figure()
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + surf_name + f'_2dheat.{format}',
                                  dpi=300, bbox_inches='tight', format=format)

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(surf_file + '_' + surf_name + f'_3dsurface.{format}', dpi=300,
                bbox_inches='tight', format=format)

    f.close()
    if show:
        plt.show()


def plot_trajectory(proj_file, dir_file, format='pdf', show=False):
    """ Plot optimization trajectory on the plane spanned by given directions."""

    assert exists(proj_file), 'Projection file does not exist.'
    f = np.load(proj_file, allow_pickle=True).flat[0]
    fig = plt.figure()
    plt.plot(f['proj_xcoord'], f['proj_ycoord'], marker='.')
    plt.tick_params('y', labelsize='x-large')
    plt.tick_params('x', labelsize='x-large')

    if exists(dir_file):
        f2 = h5py.File(dir_file,'r')
        if 'explained_variance_ratio_' in f2.keys():
            ratio_x = f2['explained_variance_ratio_'][0]
            ratio_y = f2['explained_variance_ratio_'][1]
            plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
            plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
        f2.close()

    fig.savefig(proj_file + f'.{format}', dpi=300, bbox_inches='tight', format=format)
    if show:
        plt.show()


def plot_contour_trajectory(surf_file, dir_file, proj_file, surf_name='loss_vals',
                            vmin=0.1, vmax=10, vlevel=0.5, format='pdf', show=False):
    """2D contour + trajectory"""

    assert exists(surf_file) and exists(proj_file) and exists(dir_file)

    # plot contours
    f = h5py.File(surf_file,'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)
    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])

    fig = plt.figure()
    CS1 = plt.contour(X, Y, Z, levels=np.arange(vmin, vmax, vlevel))
    CS2 = plt.contour(X, Y, Z, levels=np.logspace(1, 8, num=8))

    # plot trajectories
    pf = np.load(proj_file, allow_pickle=True).flat[0]
    plt.plot(pf['proj_xcoord'], pf['proj_ycoord'], marker='.')

    # plot red points when learning rate decays
    # for e in [150, 225, 275]:
    #     plt.plot([pf['proj_xcoord'][e]], [pf['proj_ycoord'][e]], marker='.', color='r')

    # add PCA notes
    df = h5py.File(dir_file,'r')
    ratio_x = df['explained_variance_ratio_'][0]
    ratio_y = df['explained_variance_ratio_'][1]
    plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
    plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
    df.close()
    plt.clabel(CS1, inline=1, fontsize=6)
    plt.clabel(CS2, inline=1, fontsize=6)
    fig.savefig(proj_file + '_' + surf_name + f'_2dcontour_proj.{format}', dpi=300,
                bbox_inches='tight', format=format)
    if show:
        plt.show()


def plot_2d_eig_ratio(surf_file, val_1='min_eig', val_2='max_eig',
                      format='pdf', show=False):
    """ Plot the heatmap of eigenvalue ratios, i.e., |min_eig/max_eig| of hessian """

    print('------------------------------------------------------------------')
    print('plot_2d_eig_ratio')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    f = h5py.File(surf_file,'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    Z1 = np.array(f[val_1][:])
    Z2 = np.array(f[val_2][:])

    # Plot 2D heatmaps with color bar using seaborn
    abs_ratio = np.absolute(np.divide(Z1, Z2))
    print(abs_ratio)

    fig = plt.figure()
    sns_plot = sns.heatmap(abs_ratio, cmap='viridis', vmin=0, vmax=.5, cbar=True,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + val_1 + '_' + val_2 + f'_abs_ratio_heat_sns.{format}',
                                  dpi=300, bbox_inches='tight', format=format)

    # Plot 2D heatmaps with color bar using seaborn
    ratio = np.divide(Z1, Z2)
    print(ratio)
    fig = plt.figure()
    sns_plot = sns.heatmap(
        ratio, cmap='viridis', cbar=True, xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + val_1 + '_' + val_2 + f'_ratio_heat_sns.{format}',
                                  dpi=300, bbox_inches='tight', format=format)
    f.close()
    if show:
        plt.show()



""" https://github.com/tomgoldstein/loss-landscape/scheduler.py
    A task scheduler that assign unfinished jobs to different workers.
"""

def get_unplotted_indices(vals, xcoordinates, ycoordinates=None):
    """
    Args:
      vals: values at (x, y), with value -1 when the value is not yet calculated.
      xcoordinates: x locations, i.e.,[-1, -0.5, 0, 0.5, 1]
      ycoordinates: y locations, i.e.,[-1, -0.5, 0, 0.5, 1]

    Returns:
      - a list of indices into vals for points that have not yet been calculated.
      - a list of corresponding coordinates, with one x/y coordinate per row.
    """

    # Create a list of indices into the vectorizes vals
    inds = np.array(range(vals.size))

    # Select the indices of the un-recorded entries, assuming un-recorded entries
    # will be smaller than zero. In case some vals (other than loss values) are
    # negative and those indexces will be selected again and calcualted over and over.
    inds = inds[vals.ravel() <= 0]

    # Make lists containing the x- and y-coodinates of the points to be plotted
    if ycoordinates is not None:
        # If the plot is 2D, then use meshgrid to enumerate all coordinates in the 2D mesh
        xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
        s1 = xcoord_mesh.ravel()[inds]
        s2 = ycoord_mesh.ravel()[inds]
        return inds, np.c_[s1,s2]
    else:
        return inds, xcoordinates.ravel()[inds]


def split_inds(num_inds, nproc=1):
    """
    Evenly slice out a set of jobs that are handled by each MPI process.
      - Assuming each job takes the same amount of time.
      - Each process handles an (approx) equal size slice of jobs.
      - If the number of processes is larger than rows to divide up, then some
        high-rank processes will receive an empty slice rows, e.g., there will be
        3, 2, 2, 2 jobs assigned to rank0, rank1, rank2, rank3 given 9 jobs with 4
        MPI processes.
    """

    chunk = num_inds // nproc
    remainder = num_inds % nproc
    splitted_idx = []
    for rank in range(0, nproc):
        # Set the starting index for this slice
        start_idx = rank * chunk + min(rank, remainder)
        # The stopping index can't go beyond the end of the array
        stop_idx = start_idx + chunk + (rank < remainder)
        splitted_idx.append(range(start_idx, stop_idx))

    return splitted_idx


def get_job_indices(vals, xcoordinates, ycoordinates, comm=None):
    """ Prepare the job indices over which coordinate to calculate.

    Args:
        vals: the value matrix
        xcoordinates: x locations, i.e.,[-1, -0.5, 0, 0.5, 1]
        ycoordinates: y locations, i.e.,[-1, -0.5, 0, 0.5, 1]
        comm: MPI environment

    Returns:
        inds: indices that splitted for current rank
        coords: coordinates for current rank
        inds_nums: max number of indices for all ranks
    """

    inds, coords = get_unplotted_indices(vals, xcoordinates, ycoordinates)

    rank = 0 if comm is None else comm.Get_rank()
    nproc = 1 if comm is None else comm.Get_size()
    splitted_idx = split_inds(len(inds), nproc)

    # Split the indices over the available MPI processes
    inds = inds[splitted_idx[rank]]
    coords = coords[splitted_idx[rank]]

    # Figure out the number of jobs that each MPI process needs to calculate.
    inds_nums = [len(idx) for idx in splitted_idx]

    return inds, coords, inds_nums
