#include "signup_window.h"
#include "ui_signup_window.h"
#include <QPushButton>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QLabel>
#include "mainwindow.h"
signUp_window::signUp_window(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::signUp_window)
{
    ui->setupUi(this);

    // Create main layout
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Initialize camera components
    camera = new QCamera(this);
    viewfinder = new QVideoWidget(this);
    session = new QMediaCaptureSession(this);
    imageCapture = new QImageCapture(this);

    // Create UI elements
    userNameEdit = new QLineEdit(this);
    QPushButton *snapButton = new QPushButton("Take Photo", this);

    // Configure camera session
    session->setCamera(camera);
    session->setVideoOutput(viewfinder);
    session->setImageCapture(imageCapture);

    // Add widgets to layout
    mainLayout->addWidget(viewfinder);
    mainLayout->addWidget(new QLabel("Name:", this));  // Optional label
    mainLayout->addWidget(userNameEdit);
    mainLayout->addWidget(snapButton);

    // Set the layout on the dialog
    this->setLayout(mainLayout);

    // Connect signals
    connect(snapButton, &QPushButton::clicked, [this]() {
        imageCapture->capture();
    });
    connect(imageCapture, &QImageCapture::imageCaptured,
            this, &signUp_window::onImageCaptured);

    // Start camera
    camera->start();
}

signUp_window::~signUp_window()
{
    delete ui;
}

void signUp_window::onImageCaptured(int id, const QImage &preview)
{
    Q_UNUSED(id);

    // Save or process the image
    QString filename = "captured_image.jpg";
    preview.save(filename);
    qDebug() << "Image saved to:" << filename;

   QString userName=userNameEdit->text();
  // Run the Python script
    QProcess *process = new QProcess(this);
    process->start("python", QStringList() << "newEmbedding.py" << filename
                                           << userName);

    // Optional: connect to signals for finished or error output
    connect(process, &QProcess::readyReadStandardOutput, [=]() {
        qDebug() << "Output:" << process->readAllStandardOutput();
    });
    connect(process, &QProcess::readyReadStandardError, [=]() {qDebug() << "Error:" << process->readAllStandardError();
    });
}

void signUp_window::on_pushButton_clicked()
{
    MainWindow* O= new MainWindow();
    this->hide();
    O->show();
}

