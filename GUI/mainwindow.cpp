#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "signup_window.h"
#include "loginwindow.h"
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{

    ui->setupUi(this);

    this->setAutoFillBackground(true);
    QPixmap bg(":/main/build/Desktop_Qt_6_9_0_MinGW_64_bit-Debug/backgrounds/main.png");
    bg = bg.scaled(this->size(), Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);

    QPalette palette;
    palette.setBrush(QPalette::Window, QBrush(bg));
    this->setPalette(palette);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_sign_up_push_button_clicked()
{
    signUp_window* O= new signUp_window();
    this->hide();
    O->show();

}


void MainWindow::on_login_push_button_clicked()
{
    loginWindow* O= new loginWindow();
    this->hide();
    O->show();
}

