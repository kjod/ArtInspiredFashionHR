from options.test_options import TestOptions
from analysis.visualise import display_batch, create_training_image, create_sample_iterations_over_models, create_samples_of_model
from analysis.fid_score import fid_score
import os
import datetime

SAVE_PATH = 'analysis/saved_figures/'

def get_latest(path):
    files = os.listdir(path)
    files.sort(key=lambda date: datetime.datetime.strptime(date, '%Y.%m.%d-%H.%M.%S'), reverse=True)
    run = files[0]
    return run

def get_labels(which_direction):
    AtoB = which_direction == 'AtoB'
    fake_labels = ['fake_B', 'fake_A'] if AtoB else ['fake_A', 'fake_B']
    real_labels = ['real_A', 'real_B'] if AtoB else ['real_B', 'real_A']
    return fake_labels, real_labels

def main():
    test_opt = TestOptions()
    args = test_opt.parse_options()

    save_fig = False
    if args.save_fig:
        print('All figures are saved in analysis/saved_figures')
        os.makedirs(SAVE_PATH, exist_ok=True)
        save_fig = True

    print(args)

    ### Get paths for desired modesls ###
    path = os.path.join(args.location_model_dir, args.model, 'logs', args.dataset)

    if not os.path.exists(path):
        print('No logs at %s' % (path))
        return

    ### Get latest run if not already provided ###
    if args.run == 'latest':
        run = get_latest(path)
        print('Using latest run %s ' % (args.model))
    else:
        run = args.run

    path = os.path.join(path, run)

    if not os.path.exists(path):
        print('Run does not exists %s' % (run))
        return
    else:
        print('Path of run %s' % path )


    ### Quantitative tests ###
    if args.fid_test:
        model_path = os.path.join(path, 'model')
        paths = [os.path.join(path, 'samples', 'individual'), os.path.join('data', 'redbubble', 'images')]
        model_param_loc = os.path.join(path, 'model_params.json')
        fid_score(paths, args, model_param_loc, model_path)

    ### Qualitative visuals ###
    if args.display_batch:
        display_batch(os.path.join(path, 'samples', 'batched'), save_fig, SAVE_PATH)

    if args.create_model_training_image:
        #Automate this
        fake_labels, real_labels = get_labels(args.which_direction)
        create_training_image(os.path.join(path, 'samples', 'individual'), args.n_cols, args.n_rows, save_fig, SAVE_PATH, real_labels, fake_labels, args.type, args.saved_name)

    if args.create_models_overview:
        fake_labels, real_labels = get_labels(args.which_direction)
        models = [args.model] + args.comparsion_models
        model_paths = [os.path.join(path, 'samples', 'individual')]
        #Get other paths
        if args.list_comparison_runs:
            print('Todo')
        else:
            for m in args.comparsion_models:
                m_path = os.path.join(args.location_model_dir, m, 'logs', args.dataset)
                run = get_latest(m_path)
                model_paths.append(os.path.join(m_path, run, 'samples', 'individual'))
        create_sample_iterations_over_models(models, model_paths, args.n_cols, args.n_rows, save_fig, SAVE_PATH, real_labels, 
                                             fake_labels, args.saved_name, args.example_no)

    if args.create_samples_overview:
        model_param_loc = os.path.join(path, 'model_params.json')
        model_path = os.path.join(path, 'model')
        create_samples_of_model(args.n_cols, args.n_rows, save_fig, SAVE_PATH, args.type, args.saved_name, args.model_iter, 
            model_param_loc, model_path)

    if args.check_stats:
        import subprocess
        print(os.path.join(path, 'results'))
        print(subprocess.check_output(['tensorboard', '--logdir', os.path.join(path, 'results')]))

if __name__ == '__main__':
    main()