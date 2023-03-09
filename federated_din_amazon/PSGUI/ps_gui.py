#from PyQt5 import QtCore, QtGui, QtWidgets
import PyQt5 as Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph as pg
import csv
import sys
import time


# main window
class MainUi(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_round = 0
        self.current_round_elapsed_time = 0
        self.last_round = 0
        self.best_round = 0
        self.best_round_auc = 0.0
        self.best_round_loss = 100.0
        self.round_list = []
        self.worker_num = 0

    def setup_ui(self):
        self.setWindowTitle("Parameter Server GUI") # title
        pg.setConfigOption("background", 'w')
        pg.setConfigOption("foreground", 'k')
        self.main_widget = QWidget() # create a widget
        self.main_layout = QGridLayout() # create a grid layout
        self.setCentralWidget(self.main_widget)
        self.main_widget.setLayout(self.main_layout)

        self.title_style = {"size": "16pt"}
        self.axis_label_style = {"color": "FFFFFF", "font-size": "14pt"}
        self.axis_tick_style = QFont('Arial', 11)
        self.axis_width = 70
        self.text_style = QFont('Arial', 14)

        #-------------------- plot_zone 1 AUC --------------------#
        self.plot_widget_auc = pg.PlotWidget()
        self.plot_widget_auc.showGrid(x = True, y = True, alpha = 0.2)
        self.plot_widget_auc.setTitle(title = "AUC Over Round Num", **self.title_style)
        self.plot_widget_auc.getAxis("left").setLabel(text = "AUC", **self.axis_label_style)
        self.plot_widget_auc.getAxis("left").setWidth(w = self.axis_width)
        self.plot_widget_auc.getAxis("left").setStyle(tickFont = self.axis_tick_style, textFillLimits = [(0, 0.4)])
        self.plot_widget_auc.getAxis("bottom").setLabel(text = "Round Num", **self.axis_label_style)
        self.plot_widget_auc.getAxis("bottom").setStyle(tickFont = self.axis_tick_style, textFillLimits = [(0, 0.4)])
        self.main_layout.addWidget(self.plot_widget_auc, 0, 0, 4, 4)
        self.plot_widget_auc_plot = self.plot_widget_auc.plot()
        #self.plot_widget_auc.setYRange(max = 0.65, min = 0.5)
        self.auc_list = []

        #-------------------- plot_zone 2 LOSS --------------------#
        self.plot_widget_loss = pg.PlotWidget()
        self.plot_widget_loss.showGrid(x = True, y = True, alpha = 0.2)
        self.plot_widget_loss.setTitle(title = "Loss Over Round Num", **self.title_style)
        self.plot_widget_loss.getAxis("left").setLabel(text = "Loss", **self.axis_label_style)
        #self.plot_widget_loss.getAxis("left").setWidth(w = self.axis_width)
        self.plot_widget_loss.getAxis("left").setStyle(tickFont = self.axis_tick_style, textFillLimits = [(0, 0.4)])
        self.plot_widget_loss.getAxis("bottom").setLabel(text = "Round Num", **self.axis_label_style)
        self.plot_widget_loss.getAxis("bottom").setStyle(tickFont = self.axis_tick_style, textFillLimits = [(0, 0.4)])
        self.main_layout.addWidget(self.plot_widget_loss, 0, 4, 4, 4)
        self.plot_widget_loss_plot = self.plot_widget_loss.plot()
        #self.plot_widget_loss.setYRange(max = 0.4, min = 0)
        self.loss_list = []

        #-------------------- information --------------------#
        self.setStyleSheet("QLabel{background:white;}"
                           "QLabel{color:#000000;font-size:14pt;font-weight:regular;font-family:Arial;}"
                           "QLabel:hover{color:999999;}")
        self.label_last_round = QLabel()
        self.main_layout.addWidget(self.label_last_round, 4, 0, 1, 4)
        self.label_last_round_auc = QLabel()
        self.main_layout.addWidget(self.label_last_round_auc, 4, 4, 1, 2)
        self.label_last_round_loss = QLabel()
        self.main_layout.addWidget(self.label_last_round_loss, 4, 6, 1, 2)
        self.label_best_round = QLabel()
        self.main_layout.addWidget(self.label_best_round, 5, 0, 1, 4)
        self.label_best_round_auc = QLabel()
        self.main_layout.addWidget(self.label_best_round_auc, 5, 4, 1, 2)
        self.label_best_round_loss = QLabel()
        self.main_layout.addWidget(self.label_best_round_loss, 5, 6, 1, 2)

        #-------------------- progress_bar --------------------#
        self.progress_widget = QFrame()
        self.progress_widget.setFrameShape(QFrame.StyledPanel | QFrame.Plain)
        #self.progress_widget.setLineWidth(3)
        self.progress_layout = QGridLayout()
        self.progress_widget.setLayout(self.progress_layout)
        self.label_progress = QLabel()
        self.label_progress.setStyleSheet("background-color:rgba(0,0,0,0%)");
        self.setup_current_round_elapsed_timer()
        self.progress_layout.addWidget(self.label_progress, 0, 0, 1, 8)
        self.setup_progress_bar_distribute()
        self.progress_layout.addWidget(self.progress_bar_distribute, 1, 0, 1, 4)
        self.setup_progress_bar_collect()
        self.progress_layout.addWidget(self.progress_bar_collect, 1, 4, 1, 4)
        self.setup_progress_bar_busy()
        self.progress_layout.addWidget(self.progress_bar_busy, 2, 0, 1, 8)
        self.progress_bar_busy_text = QLabel()
        self.progress_bar_busy_text.setText(" ")
        self.progress_bar_busy_text.setAlignment(Qt.AlignCenter)
        self.progress_bar_busy_text.setStyleSheet("font-size:14pt;font-weight:regular;font-family:Arial;background-color:rgba(0,0,0,0%)");
        self.progress_layout.addWidget(self.progress_bar_busy_text, 2, 0, 1, 8)
        self.main_layout.addWidget(self.progress_widget, 6, 0, 3, 8)

    def setup_data_update_worker(self, WorkerClass):
        self.thread_data_update_worker = WorkerClass()
        self.thread_data_update_worker.data_update_value.connect(self.update_data_list)
        self.thread_data_update_worker.current_round_value.connect(self.update_label_progress)
        self.thread_data_update_worker.progress_bar_busy_state_value.connect(self.update_progress_bar_busy)
        self.thread_data_update_worker.progress_bar_distribute_value.connect(self.update_progress_bar_distribute)
        self.thread_data_update_worker.progress_bar_collect_value.connect(self.update_progress_bar_collect)
        self.thread_data_update_worker.start()

    def set_worker_num(self, value):
        self.worker_num = value
        self.progress_bar_distribute.setMaximum(value)
        self.progress_bar_collect.setMaximum(value)
    
    def refresh(self):
        pg.QtGui.QApplication.processEvents()

    def setup_progress_bar_distribute(self):
        self.progress_bar_distribute = QProgressBar(self.main_widget)
        self.progress_bar_distribute.setFormat("Distributing Models: %p%")
        self.progress_bar_distribute.setFont(self.text_style) 

    def setup_progress_bar_collect(self):
        self.progress_bar_collect = QProgressBar(self.main_widget)
        self.progress_bar_collect.setFormat("Collecting Updates: %p%")
        self.progress_bar_collect.setFont(self.text_style) 
        #self.thread_progress_bar = ProgressBarWorker()
        #self.thread_progress_bar.progress_bar_value.connect(self.update_progress_bar)
        #self.thread_progress_bar.start()

    def setup_progress_bar_busy(self):
        self.progress_bar_busy = QProgressBar(self.main_widget)
        self.progress_bar_busy.setFormat("Evaluating...")
        self.progress_bar_busy.setFont(self.text_style)
        self.progress_bar_busy.setMinimum(0)
        self.progress_bar_busy.setMaximum(1)
    
    def update_progress_bar_distribute(self, value):
        self.progress_bar_distribute.setValue(value)

    def update_progress_bar_collect(self, value):
        self.progress_bar_collect.setValue(value)

    def update_progress_bar_busy(self, state, round_num):
        if state == 1:
            self.progress_bar_busy.setMaximum(0)
            self.progress_bar_busy_text.setText("<font color=white>Round {:d} Model Evaluating...</font>".format(round_num))
        else:
            self.progress_bar_busy.setMaximum(1)
            self.progress_bar_busy_text.setText("<font color=black>Round {:d} Model Evaluated.</font>".format(round_num))

    def append_auc(self, value):
        self.auc_list.append(value)
        """
        if (self.last_round > 50):
            if (self.last_round % 10 != 0):
                return
            else:
                self.plot_widget_auc_plot.setData(x = self.round_list[::10], y = self.auc_list[::10], pen = pg.mkPen(color = (54, 130, 190), width = 3))
        else:
        """
        self.plot_widget_auc_plot.setData(x = self.round_list, y = self.auc_list, pen = pg.mkPen(color = (54, 130, 190), width = 3))

    def append_loss(self, value):
        self.loss_list.append(value)
        """
        if (self.last_round > 50):
            if (self.last_round % 10 != 0):
                return
            else:
                self.plot_widget_loss_plot.setData(x = self.round_list[::10], y = self.loss_list[::10], pen = pg.mkPen(color = (69, 167, 118), width = 3))
        else:
        """
        self.plot_widget_loss_plot.setData(x = self.round_list, y = self.loss_list, pen = pg.mkPen(color = (69, 167, 118), width = 3))

    def append_round_auc_loss(self, round_num, auc, loss):
        self.last_round = round_num
        self.round_list.append(round_num)
        self.append_auc(auc)
        self.append_loss(loss)
        #self.update_label_progress(round_num + 1)
        self.update_label_last_round(round_num)
        self.update_label_last_round_auc(auc)
        self.update_label_last_round_loss(loss)
        if auc >= self.best_round_auc:
            self.best_round = round_num
            self.best_round_auc = auc
            self.best_round_loss = loss
            self.update_label_best_round(self.best_round)
            self.update_label_best_round_auc(self.best_round_auc)
            self.update_label_best_round_loss(self.best_round_loss)

    def update_data_list(self, data):
        self.append_round_auc_loss(data[0], data[1], data[2])
        #self.update_progress_bar_distribute(data[3] / data[4] * 100)
        #self.update_progress_bar_collect(data[3] / data[4] * 50)

    def update_label_progress(self, current_round):
        self.current_round = current_round
        self.reset_current_round_elapsed_timer()
        self.label_progress.setText("<p align='center'>Current Round: <b><u>{:d}</u></b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Elapsed Time: {:d}s</p>".format(self.current_round, self.current_round_elapsed_time))

    def refresh_label_progress(self):
        self.label_progress.setText("<p align='center'>Current Round: <b><u>{:d}</u></b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Elapsed Time: {:d}s</p>".format(self.current_round, self.current_round_elapsed_time))
        self.refresh()

    def increase_elapsed_time(self):
        self.current_round_elapsed_time += 1
        self.refresh_label_progress()

    def setup_current_round_elapsed_timer(self):
        self.current_round_elapsed_timer = QTimer(self)
        self.current_round_elapsed_timer.timeout.connect(self.increase_elapsed_time)
        self.current_round_elapsed_timer.start(1000)

    def reset_current_round_elapsed_timer(self):
        self.current_round_elapsed_timer.stop()
        self.current_round_elapsed_time = 0
        self.current_round_elapsed_timer.start(1000)

    def update_label_last_round(self, value):
        self.label_last_round.setText("<table width=\"100%\"><td width=\"50%\" align='left'>Last Round: </td><td width=\"50%\" align='right'><b><u>{:d}</u></b></td></table>".format(value))

    def update_label_last_round_auc(self, value):
        self.label_last_round_auc.setText("<table width=\"100%\"><td width=\"50%\" align='left'>AUC: </td><td width=\"50%\" align='right'>{:.6f}</td></table>".format(value))

    def update_label_last_round_loss(self, value):
        self.label_last_round_loss.setText("<table width=\"100%\"><td width=\"50%\" align='left'>Loss: </td><td width=\"50%\" align='right'>{:.6f}</td></table>".format(value))
    
    def update_label_best_round(self, value):
        self.label_best_round.setText("<table width=\"100%\"><td width=\"50%\" align='left'>Best Round: </td><td width=\"50%\" align='right'><b><u>{:d}</u></b></td></table>".format(value))

    def update_label_best_round_auc(self, value):
        self.label_best_round_auc.setText("<table width=\"100%\"><td width=\"50%\" align='left'>AUC: </td><td width=\"50%\" align='right'>{:.6f}</td></table>".format(value))

    def update_label_best_round_loss(self, value):
        self.label_best_round_loss.setText("<table width=\"100%\"><td width=\"50%\" align='left'>Loss: </td><td width=\"50%\" align='right'>{:.6f}</td></table>".format(value))


class _DataUpdateWorker(QThread):
    data_update_value = pyqtSignal(list)
    current_round_value = pyqtSignal(int)
    progress_bar_busy_state_value = pyqtSignal(int, int)
    progress_bar_distribute_value = pyqtSignal(int)
    progress_bar_collect_value = pyqtSignal(int)
    def __init__(self):
        super(_DataUpdateWorker, self).__init__()

    def run(self):
        csv_in = open("results.csv", "r")
        reader = csv.reader(csv_in)

        row_count = sum(1 for line in open("results.csv", "r")) - 1
        self.progress_bar_busy_state_value.emit(1, 0)
        for item in reader:
            if reader.line_num == 1:
                continue
            time.sleep(0.5)
            self.data_update_value.emit([int(item[0]), float(item[1]), float(item[2])])
            self.current_round_value.emit(reader.line_num)
            self.progress_bar_distribute_value.emit(reader.line_num)
            self.progress_bar_collect_value.emit(reader.line_num)


def main():
    app = QApplication(sys.argv)
    gui = MainUi()
    gui.setup_ui()
    gui.setup_data_update_worker(_DataUpdateWorker)
    gui.set_worker_num(100)
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()