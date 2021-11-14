import os


def create_class_res_directory(sub_directory):
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)
    if not os.path.exists(os.path.join(sub_directory,"plots")):
        os.makedirs(os.path.join(sub_directory,"plots"))


def save_loss_file(_file, itr, dloss, gloss):
    _file.write("%d,%f,%f\n"%(itr,dloss,gloss))


def create_loss_file(sub_directory):
    file = open(os.path.join(sub_directory, 'loss_logs.csv'), 'w')
    file.write('Iteration,Discriminator Loss,Generator Loss\n')
    return file