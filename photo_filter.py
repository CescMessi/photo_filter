import numpy as np
import ncnn
import shutil
import sys
import os
import cv2
# 避免OpenCV的Qt与PyQt冲突
for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

from gui import Ui_Form
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, QSettings, QObject, QTranslator
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QGraphicsPixmapItem, QGraphicsScene
from PyQt5 import QtCore


# 针对Windows中文路径做特殊处理
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(
        file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img


class MyMainForm(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)

        self.result_dict = dict()
        self.infer_thread = None
        self.copy_file_thread = None

        self.settings = QSettings('config.ini', QSettings.IniFormat)
        pic_path = self.settings.value('path/pic_path', '')
        out_path = self.settings.value('path/output_path', '')
        self.inputtextEdit.setText(pic_path)
        self.outputtextEdit.setText(out_path)

        self.listWidget.addItem('分数\t文件名')

        self.listWidget.itemClicked.connect(self.list_clicked)
        self.picButton.clicked.connect(self.select_pic_folder)
        self.outButton.clicked.connect(self.select_out_folder)
        self.startButton.clicked.connect(self.start)

    def select_pic_folder(self):
        # 选择输入图片文件夹
        path = self.settings.value('path/pic_path', '')
        foldername = QFileDialog.getExistingDirectory(
            self, "Select Directory", path)
        self.inputtextEdit.setText(foldername)
        self.settings.setValue('path/pic_path', foldername)

    def select_out_folder(self):
        # 选择输出图片文件夹
        path = self.settings.value('path/output_path', '')
        foldername = QFileDialog.getExistingDirectory(
            self, "Select Directory", path)
        self.outputtextEdit.setText(foldername)
        self.settings.setValue('path/output_path', foldername)

    def list_clicked(self, item):
        # 点击listview展示文件名对应的图片
        img_name = item.text().split('\t')[1]
        if img_name == '文件名':
            return
        dir_name = self.inputtextEdit.toPlainText()
        img_path = os.path.join(dir_name, img_name)

        self.graphicsView.scene_img = QGraphicsScene()
        imgShow = QPixmap()
        imgShow.load(img_path)
        imgShowItem = QGraphicsPixmapItem()
        imgShowItem.setPixmap(QPixmap(imgShow))
        self.graphicsView.scene_img.addItem(imgShowItem)
        self.graphicsView.setScene(self.graphicsView.scene_img)
        self.graphicsView.fitInView(QGraphicsPixmapItem(
            QPixmap(imgShow)), mode=QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def start(self):
        # 开始使用模型进行评分
        dir_path = self.inputtextEdit.toPlainText()
        param_path = self.settings.value('model/param', 'models/model.ncnn.param')
        bin_path = self.settings.value('model/bin', 'models/model.ncnn.bin')
        self.listWidget.clear()
        self.listWidget.addItem('分数\t文件名')
        self.infer_thread = QThread(parent=self)
        self.infer = Inference(dir_path, param_path, bin_path)
        self.infer.moveToThread(self.infer_thread)
        self.infer_thread.started.connect(self.infer.run)
        self.infer.result_signal.connect(self.update_result)
        self.infer.percent_signal.connect(self.update_pbar)
        self.infer.done_signal.connect(self.after_infer)
        
        self.infer_thread.start()

    def after_infer(self):
        # 模型评分完成，复制筛选的文件到输出文件夹
        self.textBrowser.append('筛选图片复制中...')
        self.copy_file = CopyFile(
            self.inputtextEdit.toPlainText(),
            self.outputtextEdit.toPlainText(),
            self.result_dict,
            self.spinBox.value()
        )
        
        self.copy_file_thread = QThread(parent=self)
        self.copy_file.moveToThread(self.copy_file_thread)
        self.copy_file_thread.started.connect(self.copy_file.run)
        self.copy_file.result_signal.connect(self.update_file_log)
        self.copy_file.done_signal.connect(self.all_done)
        self.copy_file_thread.start()

    def all_done(self, new_dict_list):
        # 评分与筛选、复制图片均已完成，listview展示分数文件名，点击展示图片
        self.textBrowser.append('已完成!')

        for name, score in new_dict_list:
            self.listWidget.addItem(f'{round(score, 4)}\t{name}')
        if self.listWidget.count() > 1:
            self.listWidget.item(1).setSelected(True)
            self.list_clicked(self.listWidget.item(1))

    def update_file_log(self, logstr):
        # 更新复制时的日志
        self.textBrowser.append(logstr)

    def update_pbar(self, percent):
        # 更新进度条
        self.progressBar.setValue(int(percent))

    def update_result(self, img_name, out):
        # 更新模型评分时的结果，分数为负数时img_name作为日志
        if out < 0:
            self.textBrowser.append(f'{img_name}')
            return
        self.textBrowser.append(f'{img_name}: {round(float(out),4)}分')
        self.result_dict[img_name] = out

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if self.infer_thread:
            self.infer_thread.terminate()
            self.infer_thread.quit()
        if self.copy_file_thread:
            self.copy_file_thread.terminate()
            self.copy_file_thread.quit()
        return super().closeEvent(a0)


class Inference(QObject):
    # 模型评分，使用ncnn进行推理
    result_signal = pyqtSignal(str, float)
    percent_signal = pyqtSignal(int)
    done_signal = pyqtSignal()

    def __init__(self, dir_path, param_path, bin_path):
        super(Inference, self).__init__()
        self.dir_path = dir_path
        # 模型文件路径，可自行替换模型
        self.PARAM_PATH = param_path
        self.BIN_PATH = bin_path
        self.stop = False

    def ncnn_inference(self, in0: np.ndarray):
        with ncnn.Net() as net:
            net.load_param(self.PARAM_PATH)
            net.load_model(self.BIN_PATH)

            with net.create_extractor() as ex:
                ex.input("in0", ncnn.Mat(in0).clone())
                _, out0 = ex.extract("out0")
                return np.array(out0)

    def detect_face(self, img, scaleFactor=1.1):
        # 从图像中裁切出人脸
        f_cascade = cv2.CascadeClassifier(
            'data/haarcascade_frontalface_alt.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = f_cascade.detectMultiScale(
            gray, scaleFactor=scaleFactor, minNeighbors=5)
        if len(faces) == 0:
            return cv2.resize(img, (224, 224))
        x, y, w, h = faces[0]
        cropped_image = img[y:y + h, x:x + w, :]
        resized_image = cv2.resize(cropped_image, (224, 224))
        return resized_image

    def predict(self, image_path):
        # 预测单张图片
        image = cv2.imread(image_path)
        if image is None:
            image = cv_imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = self.detect_face(image_rgb) / 255.
        face = (face - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
        face = np.transpose(face, (2, 0, 1)).astype(np.float32)
        res = self.ncnn_inference(face)
        return res

    def run(self):
        dir_path = self.dir_path
        support_suffix = ['jpg', 'jpeg', 'jpe', 'png', 'bmp']
        file_names = os.listdir(dir_path)
        result_dict = dict()
        length = len(file_names)
        for i, img_name in enumerate(file_names):
            suffix = img_name.split('.')[-1]
            if not suffix in support_suffix:
                img_name = f'后缀必须在{support_suffix}中，{img_name}读取失败!'
                out = -1
                self.result_signal.emit(img_name, float(out))
                self.percent_signal.emit(int((i+1)/float(length)*100))
                continue
            path = os.path.join(dir_path, img_name)
            out = self.predict(path)
            result_dict[img_name] = out

            self.result_signal.emit(img_name, float(out))
            self.percent_signal.emit(int((i+1)/float(length)*100))

        self.done_signal.emit()


class CopyFile(QObject):
    result_signal = pyqtSignal(str)
    done_signal = pyqtSignal(list)

    def __init__(self, dir, dst, result_dict, copy_num):
        super(CopyFile, self).__init__()
        self.dir = dir
        self.dst = dst
        self.result_dict = result_dict
        self.copy_num = copy_num

    def run(self):
        new_dict_list = sorted(self.result_dict.items(),
                               key=lambda x: x[1], reverse=True)
        copy_num = self.copy_num
        dst = self.dst
        dir = self.dir
        for ii, (name, score) in enumerate(new_dict_list[:copy_num]):
            shutil.copy(os.path.join(dir, name), dst)
            logstr = f'图片{name}分数为{round(float(score),4)}分，排名第{ii+1}，已复制到目录{dst}'
            self.result_signal.emit(logstr)
        self.done_signal.emit(new_dict_list)



if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    translator = QTranslator()
    translator.load('data/qt_zh_CN.qm')
    app.installTranslator(translator)
    myWin = MyMainForm()
    myWin.show()
    
    sys.exit(app.exec_())