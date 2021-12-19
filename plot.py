import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(plotScores, plotMeanScores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('SnakeAI Statistics')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(plotScores)
    plt.plot(plotMeanScores)
    plt.ylim(ymin=0)
    plt.text(len(plotScores)-1, plotScores[-1], str(plotScores[-1]))
    plt.text(len(plotMeanScores)-1, plotScores[-1], str(plotMeanScores[-1]))
    plt.show(block=False)
    plt.pause(.1)
