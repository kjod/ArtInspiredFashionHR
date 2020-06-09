import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import seaborn
import matplotlib.gridspec as gridspec
import os
import tkinter as tk
from matplotlib import animation
from analysis.fid_score import create_fake_data, load_model

def display_batch(path, save_fig, save_path):
    files = os.listdir(path)
    files = list(map(lambda x: int(x.replace('.png','')), files))
    files.sort()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.axis('off')

    ims = []
    print('Creating batch visualisation ...')
    for file in files:
        img = mpimg.imread(os.path.join(path, '%s.png' % (file)))
        im = ax.imshow(img, animated=True)
        text = ax.text(10, 0, 'Iteration %d' % file, fontsize=20)
        ims.append([text, im])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                    repeat_delay=800, repeat=True)

    plt.show()
    
    if save_fig:
        path = '%s%s.gif'% (save_path, 'batch_viz')
        ani.save(path, writer='imagemagick', fps=10)
        print('Batch image saved to %s' % save_path)


def get_unique_entries(path, real_label='real', train_label='train'):
    files = os.listdir(path)
    files = list(map(lambda x: x.replace('.png', ''), files))
    files = filter(lambda x: not real_label in x, files)
    files = filter(lambda x: not train_label in x, files)
    file_split = np.transpose(np.array(list(map(lambda x: x.split('_'), files))))
    unique_entries = {}

    i = 0
    for entry in file_split:
        unique_entries[i] = np.unique(entry)
        i += 1
    return unique_entries

def set_up_cols_rows(path, n_rows, n_cols, unique_entries, real_label='real'):
    sample_iterations = list(map(int, unique_entries[2]))
    sample_images = list(map(int, unique_entries[3]))

    sample_iterations.sort()
    print('Number of iterations available %d' % len(sample_iterations))
    print('n_cols requested %d' % n_cols)

    # Get iterations for rows
    if n_rows < len(sample_images):
        random.shuffle(sample_images)
        sample_iterations = sample_images[:n_rows]
    else:
        n_rows = len(sample_images)

    # Get iterations for cols
    if n_cols < len(sample_iterations):
        tmp_images = []
        if not len(sample_iterations) % n_cols == 0:
            new_length = len(sample_iterations) - (len(sample_iterations) % n_cols)
            step = new_length / n_cols 
            print('Changed length of iterations available %d' % new_length)
        else:
            step = len(sample_iterations) / n_cols
        [tmp_images.append(int(step * (i))) for i in range(n_cols)]
        sample_iterations = [sample_iterations[i] for i in tmp_images]
    else:
        n_cols = len(sample_iterations)
        print('Changed n_cols to size of iterations %d' % n_cols)

    sample_iterations.sort()
    sample_images.sort(reverse=True)

    return sample_images, sample_iterations, n_cols, n_rows


def create_training_image(path, n_cols, n_rows, save_fig, save_path, real_labels, fake_labels, type, saved_name):

    print('Creating model sample visualisation ...')
    
    sample_images, sample_iterations, n_cols, n_rows = set_up_cols_rows(path, n_rows, n_cols, get_unique_entries(path))
    n_cols += 2  # for training images
    check_type = type == 'InputTarget'

    if check_type:
        cols = ['{}'.format(sample_iterations[col]) for col in range(len(sample_iterations))]
        cols = ['Input A'] + cols
        cols.append('Target B')
        rows = ['Model {}'.format(row) for row in range(n_rows)]
        B_sample = n_cols - 1
        fake_start = 1
    else:
        cols = ['{}'.format(sample_iterations[col]) for col in range(len(sample_iterations))]
        cols = ['Domain B'] + cols
        cols = ['Domain A'] + cols
        rows = ['Model {}'.format(row) for row in range(n_rows)]
        B_sample = 1
        fake_start = 2

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 2, n_rows * 2))

    fake_label, real_labels = check_label(os.path.join(path, '%s_%s_%s.png' % (fake_labels[0], sample_iterations[0], sample_images[0])), fake_labels, real_labels)

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for row in range(len(sample_images)):
        file_name = os.path.join(path, '%s_%s.png' % (real_labels[0], sample_images[row]))
        img = mpimg.imread(file_name)
        set_axes(axes[row, 0], img)
        
        img = mpimg.imread(os.path.join(path, '%s_%s.png' % (real_labels[1], sample_images[row])))
        set_axes(axes[row, B_sample], img)
        
    #Fake
    for row in range(len(sample_images)):
        for col in range(0, len(sample_iterations)):
            img = mpimg.imread(os.path.join(path, '%s_%s_%s.png' % (fake_label, sample_iterations[col], sample_images[row])))
            set_axes(axes[row, col + fake_start], img)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.0)
    plt.show()
    plt.draw()

    if save_fig:
        save_image(save_path, saved_name, fig)        


def save_image(save_path, saved_name, fig):
    path = '%s%s.pdf'% (save_path, saved_name)
    fig.savefig(path, dpi=100)
    print('Image saved to %s' % path)

def check_combatabiltiy(models, model_paths, real_label='real'):
    
    unique_entries = get_unique_entries(model_paths[0], real_label=real_label)
    for path in model_paths:
        print(path)
        model_unique_entries = get_unique_entries(path, real_label=real_label)
        for i in unique_entries:
            unique_entries[i] = list(set(np.unique(model_unique_entries[i])) & set(unique_entries[i]))
        model_unique_entries[2] = list(map(int, model_unique_entries[2])) 
        model_unique_entries[2].sort()
        print(model_unique_entries[2])
        print('-------------------------')
    print('model_unique_entires')
    print(model_unique_entries)
    return unique_entries


def set_axes(ax, img):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.tick_params(bottom=False, left=False, labelleft=True, right=False)
    ax.grid(False)
    ax.imshow(img)
    #axes[row, col].axis('off')

def check_label(path, fake_labels, real_labels):
    label_check = os.path.exists(path)
    if label_check:##For pix2pix
        fake_label = fake_labels[0] 
    else:
        fake_label = fake_labels[1]
        real_labels = [real_labels[1], real_labels[0]] 
    return fake_label, real_labels

def create_sample_iterations_over_models(models, model_paths, n_cols, n_rows, save_fig, save_path, real_labels, fake_labels, saved_name, example_no):

    unique_entries = check_combatabiltiy(models, model_paths)
    sample_images, sample_iterations, n_cols, _ = set_up_cols_rows(os.path.join(model_paths[0]), n_rows, n_cols, unique_entries)

    cols = ['{}'.format(sample_iterations[col]) for col in range(len(sample_iterations))]
    cols.append('Target B')
    cols = ['Input A'] + cols
    rows = ['{}'.format(m) for m in models]
    n_cols +=2

    fig, axes = plt.subplots(nrows=len(models), ncols=n_cols, figsize=(n_cols * 2, len(models) * 2))

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        row = row.replace('_', ' ').replace('and', '&').replace('.',':').replace('[','(').replace(']',')')
        row = row.split(' ')
        if len(row) > 2:
            tmp = ' '.join(row[0:1]) + ' \n ' + " ".join(row[1:4])
            if len(row) > 4:
                tmp += ' \n ' + ' '.join(row[4:len(row)])
            row = tmp
        else:
            row = ' '.join(row)
        ax.set_ylabel(row, size='large')

    # Fake
    if example_no > len(sample_images):
        example_no = 0
    img_no = sample_images[example_no]
    for row in range(len(model_paths)):
        fake_label, real_labels = check_label(os.path.join(model_paths[row], '%s_%s_%s.png' % (fake_labels[0], sample_iterations[0], img_no)), fake_labels, real_labels)
        
        for col in range(0, len(sample_iterations)):
            img = mpimg.imread(os.path.join(model_paths[row], '%s_%s_%s.png' % (fake_label, sample_iterations[col], img_no)))
            set_axes(axes[row, col+1], img)
        img = mpimg.imread(os.path.join(model_paths[row], '%s_%s.png' % (real_labels[0], img_no)))
        set_axes(axes[row, 0], img)
        img = mpimg.imread(os.path.join(model_paths[row], '%s_%s.png' % (real_labels[1], img_no)))
        set_axes(axes[row, n_cols-1], img)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=.9, wspace=0.03, hspace=0.0)
    plt.show()
    plt.draw()

    if save_fig:
        save_image(save_path, saved_name, fig)   


def create_samples_of_model(n_cols, n_rows, save_fig, save_path, type, saved_name, model_iter, model_param_loc, model_path,
        fake_samples_dir='fake_test', model_name='G_A'):
    model, model_args = load_model(model_param_loc, model_path, model_iter)
    os.makedirs(fake_samples_dir, exist_ok=True)
    create_fake_data(fake_samples_dir, model, model_name, model_args, fid_score=False, reverse=False)
    fig, axes = plt.subplots(nrows=len(models), ncols=n_cols, figsize=(n_cols * 2, len(models) * 2))