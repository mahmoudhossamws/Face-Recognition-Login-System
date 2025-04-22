#ifndef SIGNUP_WINDOW_H
#define SIGNUP_WINDOW_H
#include <QWidget>
#include <QtMultimedia/QCamera>
#include <QtMultimedia/QMediaCaptureSession>
#include <QtMultimedia/QImageCapture>
#include <QtMultimediaWidgets/QVideoWidget>
#include <QDialog>
#include <QProcess>
#include <QLineEdit>

namespace Ui {
class signUp_window;
}

class signUp_window : public QDialog
{
    Q_OBJECT

public:
    explicit signUp_window(QWidget *parent = nullptr);
    ~signUp_window();

private slots:
    void onImageCaptured(int id, const QImage &preview);

    void on_pushButton_clicked();

private:
    Ui::signUp_window *ui;
    QCamera*                camera;
    QVideoWidget*           viewfinder;
    QMediaCaptureSession*   session;
    QImageCapture*          imageCapture;
    QLineEdit *userNameEdit;
    // UI
    QPushButton*            snapButton;

    // External process to run your Python script
    QProcess*               pythonProcess;
};

#endif // SIGNUP_WINDOW_H
