import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import os.path as osp

model_path = "/home/jayliu0313/mil_test/checkpoints/unet_test_contras"

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
absolute_path = os.path.join(parent_dir, model_path)
def main():
    #visualize training loss of AE
    input_path = os.path.join(absolute_path, "training_log.txt")
    save_path = os.path.join(absolute_path, "loss_info.png")
    visualize_loss(input_path, save_path) 

def visualize_loss(path, save_path):
    pretrained_info = {
        "Epoch": [],
        "Loss": [],
        "Contrastive Loss": [],
        "MSE Loss": [],
    } 
    with open(path) as f:
        for line in f.readlines():
            line = line.replace(':', '')
            s = line.split(' ')
            pretrained_info["Epoch"].append(s[1])
            pretrained_info["Loss"].append(float(s[3]))
            pretrained_info["Contrastive Loss"].append(float(s[6]))
            pretrained_info["MSE Loss"].append(float(s[9]))
    
    step_df = pd.DataFrame(pretrained_info)
    
    x_range = 40

    plt.title('Training Loss Detail') # set the title of graph
    plt.figure(figsize=(10, 7))
    axes = plt.axes()
    axes.set_ylim([0.0001, 0.01]) # set the range of y value
    plt.plot(step_df['Epoch'], step_df['Loss'], color='r')
    plt.xticks(np.arange(0, len(step_df['Epoch'])+1, x_range))
    plt.xlabel('epoch') # set the title of x axis
    plt.ylabel('loss')
    plt.savefig(save_path[0])
    plt.close()

    plt.title('Training Loss Detail') # set the title of graph
    plt.figure(figsize=(10, 7))
    axes = plt.axes()
    axes.set_ylim([0.00007, 0.002]) # set the range of y value
    plt.plot(step_df['Epoch'], step_df['Contrastive Loss'], color='g')
    plt.xticks(np.arange(0, len(step_df['Epoch'])+1, x_range))
    plt.xlabel('epoch') # set the title of x axis
    plt.ylabel('loss')
    plt.savefig(save_path[1])
    plt.close()

    plt.title('Training Loss Detail') # set the title of graph
    plt.figure(figsize=(10, 7))
    axes = plt.axes()
    axes.set_ylim([0.0001, 0.01]) # set the range of y value
    plt.plot(step_df['Epoch'], step_df['MSE Loss'], color='b')
    plt.xticks(np.arange(0, len(step_df['Epoch'])+1, x_range))
    plt.xlabel('epoch') # set the title of x axis
    plt.ylabel('loss')
    plt.savefig(save_path[2])

if __name__ == '__main__':
    main()