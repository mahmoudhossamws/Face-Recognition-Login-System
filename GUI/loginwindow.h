#ifndef LOGINWINDOW_H
#define LOGINWINDOW_H

#include <QWidget>
#include <QWidget>
#include <QtMultimedia/QCamera>
#include <QtMultimedia/QMediaCaptureSession>
#include <QtMultimedia/QImageCapture>
#include <QtMultimediaWidgets/QVideoWidget>
#include <QDialog>
#include <QProcess>
#include <QLineEdit>

namespace Ui {
class loginWindow;
}

class loginWindow : public QWidget
{
    Q_OBJECT

public:
    loginWindow(QWidget *parent = nullptr);
    ~loginWindow();
private slots:
    void onImageCaptured(int id, const QImage &preview);
    void on_pushButton_clicked();

private:
    Ui::loginWindow *ui;
    QCamera*                camera;
    QVideoWidget*           viewfinder;
    QMediaCaptureSession*   session;
    QImageCapture*          imageCapture;
    QPushButton*            snapButton;
};

#endif // LOGINWINDOW_H
