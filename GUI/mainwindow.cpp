#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "signup_window.h"
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
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

