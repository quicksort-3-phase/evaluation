import csv
import matplotlib.pyplot as plt
import sys
import mpl_toolkits.axes_grid1
import matplotlib.patches

from argparse import ArgumentParser
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from typing import List

blocks: int = 32
columns: int = 0
dataType: str = 'Double'
elemthread: int = 64
k: int = 0
threads: int = 128
threshold: int = 1
time: int = 0
total: bool = False
linear: bool = False
structs: bool = False
default_ylim = 0
default_xlim = 0

class PageSlider(Slider):

    def __init__(self, ax, label, data_array, valinit=0, valfmt='%1d',
                 closedmin=True, closedmax=True,
                 dragging=True, **kwargs):

        self.data_array = data_array
        self.facecolor=kwargs.get('facecolor',"w")
        self.activecolor = kwargs.pop('activecolor',"b")
        self.fontsize = kwargs.pop('fontsize', 10)
        self.numpages = len(self.data_array)

        super(PageSlider, self).__init__(ax, label, 0, self.numpages,
                            valinit=valinit, valfmt=valfmt, **kwargs)

        self.poly.set_visible(False)
        self.vline.set_visible(False)
        self.pageRects = []
        for i in range(self.numpages):
            facecolor = self.activecolor if i==valinit else self.facecolor
            r  = matplotlib.patches.Rectangle((float(i)/self.numpages, 0), 1./self.numpages, 1,
                                transform=ax.transAxes, facecolor=facecolor)
            ax.add_artist(r)
            self.pageRects.append(r)
            ax.text(float(i)/self.numpages+0.5/self.numpages, 0.5, str(self.data_array[i]),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=self.fontsize)
        self.valtext.set_visible(False)

    def _update(self, event):
        super(PageSlider, self)._update(event)
        self._colorize(int(self.val))

    def _colorize(self, i):
        for j in range(self.numpages):
            self.pageRects[j].set_facecolor(self.facecolor)
        self.pageRects[i].set_facecolor(self.activecolor)


def quicksort(list_to_sort: List[int]):
	'''
	Implement a totally useless quicksort, because of OCD reasons
	'''
	if(len(list_to_sort) <= 1):
		return list_to_sort
	pivot = list_to_sort[0]
	list3=[]
	list1=[]
	list2=[]
	for i in range(len(list_to_sort)):
		if(list_to_sort[i] < pivot):
			list1.append(list_to_sort[i])
		if(list_to_sort[i] > pivot):
			list2.append(list_to_sort[i])
		if(list_to_sort[i] == pivot):
			list3.append(list_to_sort[i])
	return quicksort(list1) + list3 + quicksort(list2)

def parse_csv(filename: str) -> List[List]:
	results: List[List] = []
	with open(filename) as csv_file:
		csv_reader: any = csv.reader(csv_file, delimiter=',')
		line_count: int = 0
		for row in csv_reader:
			if line_count != 0:
				results.append(normalize(row))
			line_count += 1
		print(f'Processed {line_count} lines.')
	return results

def normalize(list: any) -> list:
	return [i.strip() for i in list]

def plot(filename: str, file: str = None, format: str = "png"):
    global columns, default_ylim, time
    a_color = 'orange'
    bg_color = 'whitesmoke'
    colors = ['red', 'green', 'blue', 'violet', 'limegreen', 'lightblue', 'gold',
              'orange', 'forestgreen', 'steelblue', 'purple', 'seagreen']
    marker = ['.', '*', 'o', 'v', '^', '<', '>', '8', 'p', 'X', 'D', 'd']
    data: List[List] = parse_csv(filename)
    columns = len(data[0])
    ax_handles, ax_labels = None, None
    labels, lines, visibility = [], [], []

    if format == "emf" or format == "ps" or format == "ps" or format == "raw" or format=="rgba" or format == "svgz":
        print('Unsupported file format!')
        quit()
    
    if columns == 13:
        max_blocks = quicksort([i for i in {int(elem[0]) for elem in data}])
        max_threads = quicksort([i for i in {int(elem[1]) for elem in data}])
        elements_per_thread = quicksort([i for i in {int(elem[2]) for elem in data}])
        scan_threshold = quicksort([i for i in {int(elem[3]) for elem in data}])
        thresholds = quicksort([i for i in {int(elem[4]) for elem in data}])
        elements = quicksort([i for i in {int(elem[5]) for elem in data}])
    elif columns == 10:
        elements = quicksort([i for i in {int(elem[2]) for elem in data}])
        dataTypes = ['Double']
        pivots = quicksort([i for i in {elem[1] for elem in data}])
    elif columns == 9:
        elements = quicksort([i for i in {int(elem[1]) for elem in data}])
        algos = quicksort([i for i in {elem[0] for elem in data}])
        dataTypes = quicksort([i for i in {elem[2] for elem in data}])
    else:
        max_blocks = quicksort([i for i in {int(elem[0]) for elem in data}])
        max_threads = quicksort([i for i in {int(elem[1]) for elem in data}])
        elements_per_thread = quicksort([i for i in {int(elem[2]) for elem in data}])
        elements = quicksort([i for i in {int(elem[4]) for elem in data}])

    if file:
        data2: List[List] = parse_csv(file)
        columns2 = len(data2[0])
        if columns != columns2:
            print('Files incompatible!')
            quit()
        if columns == 13:
            elements2 = quicksort([i for i in {int(elem[5]) for elem in data2}])
        elif columns == 10:
            elements2 = quicksort([i for i in {int(elem[2]) for elem in data}])
        elif columns == 9:
            elements2 = quicksort([i for i in {int(elem[1]) for elem in data2}])
            dataTypes2 = quicksort([i for i in {elem[2] for elem in data2}])
        else:
            elements2 = quicksort([i for i in {int(elem[4]) for elem in data2}])
        lines2 = []

    fig, ax = plt.subplots(figsize=(16,9))
    plt.subplots_adjust(left=0.26, right = 0.99, bottom=0.35, top = 0.95)

    def getScanTimeValues(blocks, threads, elements, scan, thres, time, second = False):
        values = []
        if total:
            time = time + 3
        if second:
            for i in data2:
                if(int(i[0]) == blocks and int(i[1]) == threads and int(i[2]) == elements and int(i[3]) == scan and int(i[4]) == thres):
                    if linear:
                        values.append((float(i[time]) / int(i[5])) * 1000000000)
                    else:
                        values.append(float(i[time]) / int(i[5]))
        else:
            for i in data:
                if(int(i[0]) == blocks and int(i[1]) == threads and int(i[2]) == elements and int(i[3]) == scan and int(i[4]) == thres):
                    if linear:
                        values.append((float(i[time]) / int(i[5])) * 1000000000)
                    else:
                        values.append(float(i[time]) / int(i[5]))
        return values

    def getTimeValues(blocks, threads, elements, time, second = False):
        values = []
        if total:
            time = time + 3
        if second:
            for i in data2:
                if(int(i[0]) == blocks and int(i[1]) == threads and int(i[2]) == elements):
                    if linear:
                        values.append((float(i[time]) / int(i[4])) * 1000000000)
                    else:
                        values.append(float(i[time]) / int(i[4]))
        else:
            for i in data:
                if(int(i[0]) == blocks and int(i[1]) == threads and int(i[2]) == elements):
                    if linear:
                        values.append((float(i[time]) / int(i[4])) * 1000000000)
                    else:
                        values.append(float(i[time]) / int(i[4]))
        return values

    def getAlgoTimeValues(algo, dataType, time, second = False):
        values = []
        if total:
            time = time + 3
        if second:
            for i in data2:
                if(i[0] == algo and i[2] == dataType):
                    if linear:
                        values.append((float(i[time]) / int(i[1])) * 1000000000)
                    else:
                        values.append(float(i[time]) / int(i[1]))
        else:
            for i in data:
                if(i[0] == algo and i[2] == dataType):
                    if linear:
                        values.append((float(i[time]) / int(i[1])) * 1000000000)
                    else:
                        values.append(float(i[time]) / int(i[1]))
        return values

    def getPivotTimeValues(pivot, dataTypes, time, second = False):
        values = []
        if second:
            for i in data2:
                if(i[1] == pivot and i[3] == dataType):
                    if linear:
                        values.append((float(i[time]) / int(i[2])) * 1000000000)
                    else:
                        values.append(float(i[time]) / int(i[2]))
        else:
            for i in data:
                if(i[1] == pivot and i[3] == dataType):
                    if linear:
                        values.append((float(i[time]) / int(i[2])) * 1000000000)
                    else:
                        values.append(float(i[time]) / int(i[2]))
        return values

    def drawLines():
        if file:
            if columns == 13:
                for i, s in zip(lines, lines2):
                    times = getScanTimeValues(blocks, threads, elemthread, int(i.get_label()), threshold, time)
                    times2 = getScanTimeValues(blocks, threads, elemthread, int(i.get_label()), threshold, time, True)
                    i.set_ydata(times)
                    s.set_ydata(times2)
                fig.suptitle(f'Blocks: {blocks}, Threads: {threads}, Elements per Thread: {elemthread} & Threshold: {threshold}')
            elif columns == 10:
                for i, s in zip(lines, lines2):
                    times = getPivotTimeValues(i.get_label(), dataType, time)
                    times2 = getPivotTimeValues(i.get_label(), dataType, time, True)
                    if structs:
                        times.append(0)
                        times2.append(0)
                    i.set_ydata(times)
                    s.set_ydata(times2)
            elif columns == 9:
                # Algorithm plotting
                for i, s in zip(lines, lines2):
                    times = getAlgoTimeValues(i.get_label(), dataType, time)
                    times2 = getAlgoTimeValues(i.get_label(), dataType, time, True)
                    if structs:
                        times.append(0)
                        times2.append(0)
                    i.set_ydata(times)
                    s.set_ydata(times2)
                fig.suptitle(f'Data Type: {dataType}')
            else:
                for i, s in zip(lines, lines2):
                    times = getTimeValues(blocks, threads, int(i.get_label()), time)
                    times2 = getTimeValues(blocks, threads, int(i.get_label()), time, True)
                    i.set_ydata(times)
                    s.set_ydata(times2)
                fig.suptitle(f'Blocks: {blocks} & Threads: {threads}')
        else:
            if columns == 13:
                for i in lines:
                    times = getScanTimeValues(blocks, threads, elemthread, int(i.get_label()), threshold, time)
                    i.set_ydata(times)
                fig.suptitle(f'Blocks: {blocks}, Threads: {threads}, Elements per Thread: {elemthread} & Threshold: {threshold}')
            elif columns == 10:
                for i in lines:
                    times = getPivotTimeValues(i.get_label(), dataType, time)
                    i.set_ydata(times)
            elif columns == 9:
                # Algorithm plotting
                for i in lines:
                    times = getAlgoTimeValues(i.get_label(), dataType, time)
                    if structs:
                        times.append(0)
                    i.set_ydata(times)
                fig.suptitle(f'Data Type: {dataType}')
            else:
                for i in lines:
                    times = getTimeValues(blocks, threads, int(i.get_label()), time)
                    i.set_ydata(times)
                fig.suptitle(f'Blocks: {blocks} & Threads: {threads}')
        fig.canvas.draw_idle()

    def changeTime(label):
        global time
        global total
        if columns == 13:
            if label == 'Best Case':
                time = 7
            elif label == 'Worst Case':
                time = 8
            else:
                time = 9
        elif columns == 10:
            if label == 'Best Case':
                time = 4
            elif label == 'Worst Case':
                time = 5
            else:
                time = 6
        elif columns == 9:
             if label == 'Best Case':
                 time = 3
             elif label == 'Worst Case':
                 time = 4
             else:
                 time = 5
        else:
            if label == 'Best Case':
                time = 6
            elif label == 'Worst Case':
                time = 7
            else:
                time = 8
        drawLines()

    def changeTotal(label):
        global total
        if label == 'totalTime':
            total = True
        else:
            total = False
        drawLines()

    def changeScale(label):
        global linear
        ax.set_yscale(label)
        if label == 'linear':
            linear = True
        else:
            linear = False
        if linear:
            ax.set_ylabel('Time per element [ns]')
            if columns == 13:
                ax.set_xlim([pow(2, 20), elements[-1]])
                ax.set_ylim([15, 60])
            elif columns == 10:
                ax.set_xlim([pow(2, 14), elements[-1]])
                ax.set_ylim([20, 100])
            elif columns == 9:
                ax.set_xlim([pow(2, 10), elements[-1]])
                ax.set_ylim([0, 80])
            else:
                ax.set_xlim([pow(2, 16), elements[-1]])
                ax.set_ylim([30, 100])
        else:
            ax.set_ylabel('Time per element [s]')
            ax.set_yscale('log')
            ax.set_xlim([elements[0], elements[-1]])
            ax.set_ylim(default_ylim)
        drawLines()

    def update(val):
        global blocks, threads, elemthread, threshold
        blocks = max_blocks[int(psBlocks.val)]
        threads = max_threads[int(psThreads.val)]
        if columns == 13:
            elemthread = elements_per_thread[int(psElemThread.val)]
            if len(thresholds) > 1:
                threshold = thresholds[int(psThreshold.val)]
        drawLines()

    def update1(label):
        global dataType, structs
        if label == 'Record' or label == 'Vector':
            structs = True
            ax.set_xlim([1, pow(2, 26)])
        else:
            structs = False
            ax.set_xlim(default_xlim) 
        dataType = label
        if file and not label in dataTypes2:
            print('Files incompatible!')
        else: 
            drawLines()

    def update2(label):
        i = labels.index(label)
        lines[i].set_visible(not lines[i].get_visible())
        if file:
            lines2[i].set_visible(not lines2[i].get_visible())
        if ax_labels[i].startswith('_'):
            ax_labels[i] = ax_labels[i][1:]
        else:
            ax_labels[i] = '_' + ax_labels[i]
        ax.legend(ax_handles, ax_labels, loc='upper right', ncol=2, prop={'size': 11})
        plt.draw()

    def reset(event):
        psBlocks.reset()
        psThreads.reset()
        if columns == 13:
            psElemThread.reset()
            if len(thresholds) > 1:
                psThreshold.reset()

    def save(event):
        global k
        k += 1
        image = ax.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'ax_figure{k}.' + format, bbox_inches=image.expanded(1.01, 1.01))

    if columns == 13:
        time = 9
        for s, c, m in zip(scan_threshold, colors, marker):
            times = getScanTimeValues(blocks, threads, elemthread, s, threshold, time)
            if len(elements) == len(times):
                line, = plt.plot(elements, times, color=c, label=s, marker=m)
                lines.append(line)
            if file:
                times2 = getScanTimeValues(blocks, threads, elemthread, s, threshold, time, True)
                if len(elements2) == len(times2):
                    line2, = plt.plot(elements2, times2, alpha=0.5, color=c, linestyle='--')
                    lines2.append(line2)
        fig.suptitle(f'Blocks: {blocks}, Threads: {threads}, Elements per Thread: {elemthread} & Threshold: {threshold}')
    
    elif columns == 10:
        time = 6
        for p, c, m in zip(pivots, colors, marker):
            times = getPivotTimeValues(p, dataType, time)
            if len(elements) == len(times):
                line, = plt.plot(elements, times, color=c, label=p, marker=m)
                lines.append(line)
            if file:
                times2 = getPivotTimeValues(p, dataType, time)
                if len(elements2) == len(times2):
                    line2, = plt.plot(elements2, times2, alpha=0.5, color=c, linestyle='--')
                    lines2.append(line2)
        fig.suptitle(f'Pivottest')

    elif columns == 9:
        # Algorithm ploting
        time = 5
        for a, c, m in zip(algos, colors, marker):
            times = getAlgoTimeValues(a, dataType, time)
            if len(elements) == len(times):
                line, = plt.plot(elements, times, color=c, label=a, marker=m)
                lines.append(line)
            if file:
                times2 = getAlgoTimeValues(a, dataType, time, True)
                if len(elements2) == len(times2):
                    line2, = plt.plot(elements2, times2, alpha=0.5, color=c, linestyle='--')
                    lines2.append(line2)
        fig.suptitle(f'Data Type: {dataType}')

    else:
        time = 8
        for e, c, m in zip(elements_per_thread, colors, marker):
            times = getTimeValues(blocks, threads, e, time)
            if len(elements) == len(times):
                line, = plt.plot(elements, times, label=e, marker=m)
                lines.append(line)
            if file:
                times2 = getTimeValues(blocks, threads, e, time, True)
                if len(elements2) == len(times2):
                    line2, = plt.plot(elements2, times2, alpha=0.5, color=c, linestyle='--')
                    lines2.append(line2)
        fig.suptitle(f'Blocks: {blocks} & Threads: {threads}')

    
    ax_handles, ax_labels = ax.get_legend_handles_labels()
    ax.legend(ax_handles, ax_labels, loc='upper right', ncol=2, prop={'size': 11})
    ax.set_facecolor(bg_color)
    if file:
        fig.canvas.set_window_title("Compare: " + filename.replace('./results/','') + ", " + file.replace('./results/',''))
    else:
        fig.canvas.set_window_title(filename.replace('./results/',''))
    plt.grid()
    plt.tick_params(labelsize=11)
    plt.xlabel('Elements', fontsize=11)
    plt.ylabel('Time per element [s]', fontsize=11)
    plt.xlim([elements[0], elements[-1]])
    plt.xscale('log', basex=2)
    plt.yscale('log')
    
    default_ylim = ax.get_ylim()
    default_xlim = ax.get_xlim()

    xpos_axT = 0.11
    if columns == 13:
        xpos_axT = 0.01
        axElemThread = plt.axes([0.41, 0.13, 0.58, 0.04], facecolor=bg_color)
        psElemThread = PageSlider(axElemThread, 'Elements per Threads:', elements_per_thread, activecolor=a_color)
        psElemThread.on_changed(update)

        axBlocks = plt.axes([0.25, 0.23, 0.74, 0.04], facecolor=bg_color)
        psBlocks = PageSlider(axBlocks, 'Blocks:', max_blocks, activecolor=a_color)
        psBlocks.on_changed(update)

        axThreads = plt.axes([0.41, 0.18, 0.58, 0.04], facecolor=bg_color)
        psThreads = PageSlider(axThreads, 'Threads:', max_threads, activecolor=a_color)
        psThreads.on_changed(update)

        if len(thresholds) > 1:
            axThreshold = plt.axes([0.7, 0.08, 0.29, 0.04], facecolor=bg_color)
            psThreshold = PageSlider(axThreshold, 'Threshold:', thresholds, activecolor=a_color)
            psThreshold.on_changed(update)

        axReset = plt.axes([0.77, 0.028, 0.1, 0.04])
        bReset = Button(axReset, 'Reset', color=bg_color, hovercolor='0.975')
        bReset.on_clicked(reset)

    elif columns == 9:
        axDataType = plt.axes([0.01, 0.63, 0.09, 0.17], facecolor=bg_color)
        rbDataType = RadioButtons(axDataType, dataTypes, active=0, activecolor=a_color)
        rbDataType.on_clicked(update1)

        axAlgo = plt.axes([0.01, 0.05, 0.17, 0.25], facecolor=bg_color)
        labels = [line.get_label() for line in lines]
        visibility = [line.get_visible() for line in lines]
        cbAlgo = CheckButtons(axAlgo, labels, visibility)
        cbAlgo.on_clicked(update2)

    elif columns == 10:
        xpos_axT = 0.01

    else:
        xpos_axT = 0.01
        axBlocks = plt.axes([0.25, 0.2, 0.74, 0.04], facecolor=bg_color)
        psBlocks = PageSlider(axBlocks, 'Blocks:', max_blocks, activecolor=a_color)
        psBlocks.on_changed(update)

        axThreads = plt.axes([0.41, 0.15, 0.58, 0.04], facecolor=bg_color)
        psThreads = PageSlider(axThreads, 'Threads:', max_threads, activecolor=a_color)
        psThreads.on_changed(update)

        axReset = plt.axes([0.78, 0.028, 0.1, 0.04])
        bReset = Button(axReset, 'Reset', color=bg_color, hovercolor='0.975')
        bReset.on_clicked(reset)

    axSave = plt.axes([0.89, 0.028, 0.1, 0.04])
    bSave = Button(axSave, 'Save', color=bg_color, hovercolor='0.975')
    bSave.on_clicked(save)

    axYScale = plt.axes([0.01, 0.82, 0.09, 0.13], facecolor=bg_color)
    rbYScale = RadioButtons(axYScale, ('linear', 'log'), active=1, activecolor=a_color)
    rbYScale.on_clicked(changeScale)

    axTimes = plt.axes([xpos_axT, 0.63, 0.09, 0.17], facecolor=bg_color)
    rbTimes = RadioButtons(axTimes, ('Average', 'Best Case', 'Worst Case'), active=0, activecolor=a_color)
    rbTimes.on_clicked(changeTime)

    axTotal = plt.axes([0.11, 0.82, 0.09, 0.13], facecolor=bg_color)
    rbTotal = RadioButtons(axTotal, ('sortingTime', 'totalTime'), active=0, activecolor=a_color)
    rbTotal.on_clicked(changeTotal)

    #plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    '''
    Main
    '''
    # First input new results, second input old results.
    parser = ArgumentParser(
            description='Algorithm Visualizer')
    parser.add_argument('-i', '--input',
            action='store',
            default="",
            type=str,
            required=True,
            help='CSV to input',
            metavar='<file>',
            dest='file1')
    parser.add_argument('-c', '--compare',
            action='store',
            default="",
            type=str,
            required=False,
            help='CSV used for comparison',
            metavar='<file>',
            dest='file2')
    parser.add_argument('-o', '--output',
            action='store',
            default="",
            type=str,
            required=False,
            help='Output format of the plot pictures',
            metavar='<format>',
            dest='format')

    arguments = parser.parse_args()

    plot(arguments.file1, arguments.file2, arguments.format)
