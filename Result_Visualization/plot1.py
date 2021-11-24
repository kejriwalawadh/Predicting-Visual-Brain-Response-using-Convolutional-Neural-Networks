import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_score(df, network):
    plt.figure()
    df.plot(ylim=(0,20))
    plt.gcf().subplots_adjust(bottom=0.20)
    plt.xlabel('Layer')
    plt.ylabel('Noise Noramlized $R^{2}$ (%)')
    plt.title(f'{network} Performance')
    x_ticks = df.index.values.tolist()
    new_xticks=df['Layer'].tolist()
    plt.xticks(x_ticks, new_xticks, rotation=45, horizontalalignment='right')
    plt.savefig(f'{network}.png')


def plot_best(y1, y2, y3):
    x = np.arange(3)
    width = 0.2
    plt.figure()
    plt.bar(x-0.2, y1, width, color='cyan')
    plt.bar(x, y2, width, color='orange')
    plt.bar(x+0.2, y3, width, color='green')
    plt.xticks(x, ['EVC', 'IT', 'Avg'], rotation=45, horizontalalignment='right')
    plt.ylabel('Noise Noramlized $R^{2}$ (%)')
    plt.legend(["AlexNet", "ResNet", "VGG"])
    plt.title('Best Scores Comparison')
    plt.savefig('Best_Scores.png')


def main():
    with open('score.json', 'r') as fp:
        score = json.load(fp)
    
    net, layer, evc, it, avg = [], [], [], [], []
    for net_, temp1 in score.items():
        for layer_, temp2 in temp1.items():
            if layer_[0] == 'f':    #skip fully connected layers
                continue
            layer.append(layer_)
            net.append(net_)
            evc.append(temp2['EVC'])
            it.append(temp2['IT'])
            avg.append(temp2['Avg'])
            
    df = pd.DataFrame()
    df['Network'] = net
    df['Layer'] = layer
    df['EVC'] = evc
    df['IT'] = it
    df['Avg'] = avg
    
    df.sort_values(by=['Network'])
    
    df.to_csv('Scores.csv')
    
    net_list = df.Network.unique()
    for net in net_list:
        df1 = df[df['Network'] == net]
        plot_score(df1, net.upper())
    
    alexnet_best = [0]*3
    resnet_best = [0]*3
    vgg_best = [0]*3
    
    for i in df.index:
        for j in range(len(net_list)):
            if df['Network'][i]=='alexnet':
                alexnet_best[0] = max(alexnet_best[0], df['EVC'][i])
                alexnet_best[1] = max(alexnet_best[1], df['IT'][i])
                alexnet_best[2] = max(alexnet_best[2], df['Avg'][i])
            elif df['Network'][i]=='resnet':
                resnet_best[0] = max(resnet_best[0], df['EVC'][i])
                resnet_best[1] = max(resnet_best[1], df['IT'][i])
                resnet_best[2] = max(resnet_best[2], df['Avg'][i])
            elif df['Network'][i]=='vgg':
                vgg_best[0] = max(vgg_best[0], df['EVC'][i])
                vgg_best[1] = max(vgg_best[1], df['IT'][i])
                vgg_best[2] = max(vgg_best[2], df['Avg'][i])
            
    plot_best(alexnet_best, resnet_best, vgg_best)
    
    
if __name__ == '__main__':
    main()
    