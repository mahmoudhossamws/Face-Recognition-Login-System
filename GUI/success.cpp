#include "success.h"
#include "ui_success.h"
#include "mainwindow.h"
Success::Success(QWidget *parent,QString name)
    : QWidget(parent)
    , ui(new Ui::Success)
{
    ui->setupUi(this);
    ui->label->setText("welcome  "+name);



    this->setAutoFillBackground(true);
    QPixmap bg(":/main/build/Desktop_Qt_6_9_0_MinGW_64_bit-Debug/backgrounds/success.png");
    bg = bg.scaled(this->size(), Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);

    QPalette palette;
    palette.setBrush(QPalette::Window, QBrush(bg));
    this->setPalette(palette);
}

Success::~Success()
{
    delete ui;
}

void Success::on_pushButton_clicked()
{
    MainWindow* O= new MainWindow();
    this->hide();
    O->show();
}

